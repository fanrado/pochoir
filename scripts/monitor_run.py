#!/usr/bin/env python3
"""
Wrap a scripts/*.sh runner and measure it, without changing how it behaves.

The point is that a monitored run should look exactly like an unmonitored one:
the target's stdout and stderr are inherited, not captured, so its output
streams through live and in order; its exit code becomes ours; and Ctrl-C kills
the whole run rather than orphaning a GPU solve.

Usage:
    ./env/bin/python scripts/monitor_run.py --runtime --gpu-mem -- \
        ./scripts/run-task10b-drift-2cm-gap09-chamf07.sh store_foo

Everything after the first bare ``--`` is the target command, passed through
verbatim.  ``--runtime`` and ``--gpu-mem`` are INDEPENDENT opt-ins; giving
NEITHER turns both ON, so the bare form measures everything.

This is Phase 1/Step 1: the wrapper and the wall-clock measurement.  ``--gpu-mem``
is accepted and reported as requested but its sampler is a later step -- see the
note printed at exit, which says so rather than implying a measurement happened.
"""
import argparse
import csv
import datetime
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time


def _split_target(argv):
    """Split our own flags from the target command at the first bare '--'.

    argparse's REMAINDER is famously fragile about interleaved flags, and the
    target command has flags of its own that must not be interpreted here, so
    the split is done by hand before argparse ever sees the list.
    """
    try:
        cut = argv.index('--')
    except ValueError:
        return argv, []
    return argv[:cut], argv[cut + 1:]


def _parse(mine):
    ap = argparse.ArgumentParser(
        prog='monitor_run.py',
        description='Wrap a runner script and measure its runtime / GPU memory.',
        epilog="Everything after a bare '--' is the command to run.")
    ap.add_argument('--runtime', action='store_true',
                    help='measure wall-clock runtime (default when neither '
                         '--runtime nor --gpu-mem is given)')
    ap.add_argument('--gpu-mem', action='store_true',
                    help='measure GPU memory (default when neither is given)')
    ap.add_argument('--interval', type=float, default=2.0,
                    help='sampling interval in seconds (default 2.0)')
    ap.add_argument('--out-dir', default='.',
                    help='directory to write the report files into; created if '
                         'missing (default: the current directory)')
    ap.add_argument('--per-step', dest='per_step', action='store_true',
                    default=None,
                    help='attribute time and GPU memory to each `pochoir '
                         '<subcommand>` the runner echoes (default: ON when '
                         'both runtime and gpu-mem monitoring are active)')
    ap.add_argument('--no-per-step', dest='per_step', action='store_false',
                    help='disable the per-subcommand breakdown')
    ap.add_argument('--gpu-mem-scope', choices=('proc-tree', 'device'),
                    default='proc-tree',
                    help="'proc-tree' counts only the target's own processes; "
                         "'device' counts everything on the device "
                         '(default proc-tree)')
    a = ap.parse_args(mine)

    # Independent opt-ins: neither given means both on.
    if not a.runtime and not a.gpu_mem:
        a.runtime = True
        a.gpu_mem = True
    if a.interval <= 0:
        ap.error('--interval must be positive, got %r' % a.interval)
    if a.per_step is None:
        a.per_step = bool(a.runtime and a.gpu_mem)
    return a


NVIDIA_SMI = 'nvidia-smi'


def _smi(args):
    """Run one nvidia-smi query and return its stripped stdout lines.

    Raises OSError if the binary is missing and subprocess.CalledProcessError if
    it fails, so the caller decides whether that is fatal (it never is here --
    GPU sampling is best-effort and must not take the run down with it).
    """
    out = subprocess.run([NVIDIA_SMI] + args, check=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True).stdout
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def _gpu_uuid_to_index():
    """{gpu_uuid: index} so per-process rows can be bucketed by card."""
    m = {}
    for ln in _smi(['--query-gpu=index,uuid',
                    '--format=csv,noheader,nounits']):
        parts = [f.strip() for f in ln.split(',')]
        if len(parts) >= 2:
            m[parts[1]] = int(parts[0])
    return m


def _descendants(pid):
    """The full descendant PID set of `pid`, including `pid` itself.

    Walks /proc/<pid>/task/*/children, which is the only interface that reports
    children directly; ppid-scanning all of /proc would race harder and cost
    more at a 2s cadence.  A process exiting mid-walk just disappears -- its
    unreadable entry is skipped rather than failing the tick.
    """
    seen = set()
    stack = [int(pid)]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        try:
            names = os.listdir('/proc/%d/task' % cur)
        except OSError:
            continue          # exited between listing and reading
        for tid in names:
            try:
                with open('/proc/%d/task/%s/children' % (cur, tid)) as fp:
                    kids = fp.read().split()
            except OSError:
                continue
            for k in kids:
                try:
                    stack.append(int(k))
                except ValueError:
                    pass
    return seen


def _sample_proc_tree(root_pid, uuid_index):
    """Per-GPU MiB summed over the target's own processes.

    Returns {} when no descendant of the target holds GPU memory -- the caller
    treats that as "nothing of ours is on the card yet" and, per the issue,
    falls back to the whole-device reading.
    """
    mine = _descendants(root_pid)
    per_gpu = {}
    for ln in _smi(['--query-compute-apps=pid,used_gpu_memory,gpu_uuid',
                    '--format=csv,noheader,nounits']):
        parts = [f.strip() for f in ln.split(',')]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            mib = float(parts[1])
        except ValueError:
            continue          # e.g. '[N/A]' for a process we cannot see
        if pid not in mine:
            continue
        idx = uuid_index.get(parts[2])
        if idx is None:
            continue
        per_gpu[idx] = per_gpu.get(idx, 0.0) + mib
    return per_gpu


def _sample_device():
    """Per-GPU MiB for the WHOLE card, including other users' processes."""
    per_gpu = {}
    for ln in _smi(['--query-gpu=index,memory.used',
                    '--format=csv,noheader,nounits']):
        parts = [f.strip() for f in ln.split(',')]
        if len(parts) < 2:
            continue
        try:
            per_gpu[int(parts[0])] = float(parts[1])
        except ValueError:
            continue
    return per_gpu


class GpuSampler(object):
    """Daemon thread polling per-GPU memory every `interval` seconds.

    Best-effort by contract: any nvidia-smi failure disables sampling after ONE
    warning and never propagates, because a monitoring wrapper must not be able
    to kill the run it is measuring.  Samples are (elapsed_seconds, {gpu: MiB}).
    """

    def __init__(self, root_pid, interval, scope, t0_mono, log=sys.stderr):
        self.root_pid = root_pid
        self.interval = interval
        self.scope = scope
        self.t0_mono = t0_mono
        self.log = log
        self.samples = []
        self.disabled_reason = None
        self.fell_back_to_device = False
        self._stop = threading.Event()
        self._thread = None
        self._uuid_index = {}

    def start(self):
        try:
            self._uuid_index = _gpu_uuid_to_index()
        except OSError:
            self._disable('nvidia-smi not found')
            return self
        except subprocess.CalledProcessError as e:
            self._disable('nvidia-smi failed (exit %d)' % e.returncode)
            return self
        except Exception as e:                       # noqa: BLE001
            self._disable('nvidia-smi unusable (%s)' % e)
            return self
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval))

    def _disable(self, reason):
        if self.disabled_reason is None:
            self.disabled_reason = reason
            print('monitor_run: %s -- GPU memory sampling disabled for this '
                  'run (the run itself is unaffected)' % reason, file=self.log)
            self.log.flush()

    def _tick(self):
        """One reading; returns (per_gpu_mib, was_device_fallback)."""
        if self.scope == 'device':
            return _sample_device(), False
        per_gpu = _sample_proc_tree(self.root_pid, self._uuid_index)
        if not per_gpu:
            # No rows for our tree: either the solve has not touched the GPU
            # yet, or the driver will not report per-process usage to us.  The
            # whole-card reading is the documented fallback.
            #
            # IT IS NOT THE SAME MEASUREMENT: a device reading includes every
            # other process on the card, so a fallback tick can report memory
            # that is not ours at all.  Each sample records whether it was a
            # fallback so a reader can tell the two apart instead of averaging
            # them together blindly.
            self.fell_back_to_device = True
            return _sample_device(), True
        return per_gpu, False

    def _loop(self):
        while not self._stop.is_set():
            try:
                per_gpu, was_fallback = self._tick()
            except OSError:
                self._disable('nvidia-smi disappeared mid-run')
                return
            except subprocess.CalledProcessError as e:
                self._disable('nvidia-smi failed (exit %d)' % e.returncode)
                return
            except Exception as e:                   # noqa: BLE001
                self._disable('GPU sampling error (%s)' % e)
                return
            self.samples.append((round(time.monotonic() - self.t0_mono, 3),
                                 per_gpu, was_fallback))
            self._stop.wait(self.interval)

    def peaks(self, own_only=False):
        """Per-GPU peak MiB.

        `own_only` drops device-fallback ticks, which can carry other
        processes' memory; the plain peak keeps every sample.
        """
        peak = {}
        for _t, per_gpu, was_fallback in self.samples:
            if own_only and was_fallback:
                continue
            for idx, mib in per_gpu.items():
                if mib > peak.get(idx, -1.0):
                    peak[idx] = mib
        return peak


# The runner scripts' want() helper echoes the whole `pochoir ...` command line
# to stdout immediately before running it (scripts/helpers.sh:9 and the inline
# override in the run-*.sh scripts), which makes the streamed stdout a usable
# phase marker.
#
# NOTE the character class is [\w-], not \w: `induce-pixel` is one of the
# subcommands and a bare \w+ would clip it to `induce`.
_MARKER = re.compile(r'^pochoir ([\w-]+)')


class SegmentTracker(object):
    """Splits the run into segments at each `pochoir <subcommand>` marker.

    A segment runs from its marker to the next one (the last closes at the end
    of the run).  Anything echoed before the first marker is not a segment --
    it is setup, and inventing a name for it would be worse than omitting it.
    """

    def __init__(self):
        self.segments = []

    def mark(self, name, elapsed):
        if self.segments:
            self.segments[-1]['end_seconds'] = elapsed
        self.segments.append(dict(subcommand=name,
                                  start_seconds=elapsed,
                                  end_seconds=None))

    def close(self, elapsed):
        if self.segments and self.segments[-1]['end_seconds'] is None:
            self.segments[-1]['end_seconds'] = elapsed

    def finish(self, samples):
        """Attach duration and per-GPU peak over each segment's own window."""
        out = []
        for seg in self.segments:
            t0 = seg['start_seconds']
            t1 = seg['end_seconds']
            inside = [smp for smp in samples
                      if smp[0] >= t0 and (t1 is None or smp[0] < t1)]

            def _peak(rows):
                peak = {}
                for _t, per_gpu, _fb in rows:
                    for idx, mib in per_gpu.items():
                        if mib > peak.get(idx, -1.0):
                            peak[idx] = mib
                return {str(i): v for i, v in sorted(peak.items())}

            own = [smp for smp in inside if not smp[2]]
            out.append(dict(
                subcommand=seg['subcommand'],
                start_seconds=t0,
                end_seconds=t1,
                duration_seconds=(round(t1 - t0, 3)
                                  if t1 is not None else None),
                n_samples=len(inside),
                peak_mib=_peak(inside),
                # Same caveat as the run-level figure: a device-fallback tick
                # can carry other processes' memory, and on a shared card that
                # swamps the segment's real usage.  Both are reported so the
                # breakdown stays meaningful either way.
                n_samples_excluding_fallback=len(own),
                peak_mib_excluding_fallback=_peak(own),
            ))
        return out


def _pump_stdout(pipe, tracker, t0_mono, out=sys.stdout):
    """Re-emit the target's stdout line by line, tagging markers as we go.

    Parsing stdout means it has to come through a pipe rather than being
    inherited, so each line is written straight back out and flushed to keep the
    run looking live.  One consequence worth knowing: the target's stdout is no
    longer a tty, and stdout/stderr interleaving is no longer guaranteed by the
    kernel (stderr stays inherited).  That is the price of phase markers.
    """
    try:
        for raw in iter(pipe.readline, ''):
            elapsed = round(time.monotonic() - t0_mono, 3)
            out.write(raw)
            out.flush()
            m = _MARKER.match(raw.strip())
            if m and tracker is not None:
                tracker.mark(m.group(1), elapsed)
    finally:
        try:
            pipe.close()
        except Exception:                        # noqa: BLE001
            pass


def _report_stem(target, when):
    """<scriptname>_<UTC timestamp> -- the shared stem of both report files.

    The script's basename is used, with any extension dropped, so
    `./scripts/run-task10b-drift-2cm-gap09-chamf07.sh` becomes
    `run-task10b-drift-2cm-gap09-chamf07`.
    """
    name = os.path.basename(target[0]) if target else 'run'
    name = os.path.splitext(name)[0] or 'run'
    safe = ''.join(c if (c.isalnum() or c in '-_.') else '_' for c in name)
    return '%s_%s' % (safe, when.strftime('%Y%m%dT%H%M%SZ'))


def _write_csv(path, samples):
    """One row per sample per GPU: t_seconds,gpu_index,used_mib.

    Written even when there are no samples -- a header-only file says
    "measured, found nothing" and is easy to plot; a missing file is
    indistinguishable from a crash.
    """
    with open(path, 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['t_seconds', 'gpu_index', 'used_mib'])
        for t, per_gpu, _fb in samples:
            for idx in sorted(per_gpu):
                w.writerow([t, idx, per_gpu[idx]])


def _per_gpu_stats(samples):
    """{gpu_index: {peak_mib, mean_mib, n_samples}} over the given samples."""
    acc = {}
    for _t, per_gpu, _fb in samples:
        for idx, mib in per_gpu.items():
            a = acc.setdefault(idx, [0.0, 0.0, 0])   # peak, total, n
            if mib > a[0]:
                a[0] = mib
            a[1] += mib
            a[2] += 1
    return {str(i): dict(peak_mib=round(a[0], 1),
                         mean_mib=round(a[1] / a[2], 1) if a[2] else None,
                         n_samples=a[2])
            for i, a in sorted(acc.items())}


def _forward_to_group(pid):
    """Send SIGINT/SIGTERM on to the target's whole process group.

    The target is a shell script that itself spawns `pochoir` children.  Killing
    only the shell would leave a multi-GB GPU solve running with nobody waiting
    on it, so the signal goes to the process group the child leads.

    MEASURED LIMIT, one case survives SIGINT: a grandchild the runner starts
    with `&` ignores it.  POSIX requires a non-interactive shell to set SIGINT
    (and SIGQUIT) to SIG_IGN for asynchronous jobs, so the signal is delivered
    to the group and that process discards it.  Verified both ways -- a
    FOREGROUND grandchild dies on SIGINT and on SIGTERM, a backgrounded one dies
    only on SIGTERM.  Every scripts/run-*.sh solve runs in the foreground, so
    Ctrl-C does kill those cleanly; send SIGTERM if a runner ever backgrounds
    its own work.  No escalation to SIGKILL is attempted on purpose: killing a
    solve harder than asked risks a half-written store.
    """
    def handler(signum, _frame):
        try:
            os.killpg(os.getpgid(pid), signum)
        except ProcessLookupError:
            pass          # already gone; nothing to forward to
        except PermissionError:
            # Not ours to signal (should not happen for our own child), so fall
            # back to the direct child rather than failing silently.
            try:
                os.kill(pid, signum)
            except ProcessLookupError:
                pass
    return handler


def main(argv=None):
    mine, target = _split_target(list(sys.argv[1:] if argv is None else argv))
    a = _parse(mine)

    if not target:
        print("monitor_run.py: no target command -- put it after a bare '--', "
              "e.g.\n  ./env/bin/python scripts/monitor_run.py --runtime -- "
              "./scripts/run-foo.sh store_bar", file=sys.stderr)
        return 2

    out_dir = a.out_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print('monitor_run: %s' % ' '.join(target), file=sys.stderr)
    print('monitor_run: measuring%s%s (interval %.3gs, gpu-mem scope %s)'
          % (' runtime' if a.runtime else '',
             ' gpu-mem' if a.gpu_mem else '',
             a.interval, a.gpu_mem_scope), file=sys.stderr)
    sys.stderr.flush()

    # start_new_session=True puts the child in its own process group (and
    # session), which is what makes the group-wide signal forwarding above
    # possible.  stdout/stderr are deliberately NOT redirected: they are
    # inherited, so the run streams through unchanged.
    t0 = time.time()
    t0_mono = time.monotonic()
    # --per-step needs to READ stdout, so it comes through a pipe and is
    # re-emitted by _pump_stdout.  Without it stdout stays inherited, which is
    # the cheaper and more faithful path.
    tracker = SegmentTracker() if a.per_step else None
    try:
        proc = subprocess.Popen(
            target, start_new_session=True,
            stdout=(subprocess.PIPE if a.per_step else None),
            universal_newlines=True, bufsize=1)
    except FileNotFoundError:
        print('monitor_run: cannot execute %r -- no such file' % target[0],
              file=sys.stderr)
        return 127
    except PermissionError:
        print('monitor_run: cannot execute %r -- not executable' % target[0],
              file=sys.stderr)
        return 126

    pump = None
    if a.per_step:
        pump = threading.Thread(target=_pump_stdout,
                                args=(proc.stdout, tracker, t0_mono),
                                daemon=True)
        pump.start()

    sampler = None
    if a.gpu_mem:
        sampler = GpuSampler(proc.pid, a.interval, a.gpu_mem_scope,
                             t0_mono).start()

    # Report paths are fixed BEFORE the wait, so the finally block below can
    # write them no matter how the run ends.
    stem = _report_stem(target, datetime.datetime.utcnow())
    csv_path = os.path.join(out_dir, 'monitor_%s.csv' % stem)
    json_path = os.path.join(out_dir, 'monitor_%s_summary.json' % stem)

    prev = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        prev[sig] = signal.signal(sig, _forward_to_group(proc.pid))
    rc = None
    try:
        rc = proc.wait()
    finally:
        for sig, old in prev.items():
            signal.signal(sig, old)
        if pump is not None:
            # Drain whatever the target already wrote before summarising.
            pump.join(timeout=5.0)
        if sampler is not None:
            sampler.stop()
        elapsed = time.monotonic() - t0_mono
        t1 = time.time()

        # A child killed by signal N reports -N; report it the way a shell does.
        if rc is None:
            exit_code = 1                      # we never got a status at all
        elif rc >= 0:
            exit_code = rc
        else:
            exit_code = 128 + (-rc)

        samples = sampler.samples if sampler is not None else []
        stats = _per_gpu_stats(samples)

        summary = dict(
            command=target,
            exit_code=exit_code,
            wall_seconds=round(elapsed, 3),
            wall_hms=_hms(elapsed),
            started_unix=round(t0, 3),
            ended_unix=round(t1, 3),
            killed_by_signal=((-rc) if (rc is not None and rc < 0) else None),
            measured=dict(runtime=bool(a.runtime), gpu_mem=bool(a.gpu_mem)),
            interval=a.interval,
            gpu_mem_scope=a.gpu_mem_scope,
            per_gpu=stats,
            gpu_mem_note=(sampler.disabled_reason
                          if sampler is not None else None),
            fell_back_to_device=(bool(sampler.fell_back_to_device)
                                 if sampler is not None else False),
            per_step=bool(a.per_step),
        )
        if tracker is not None:
            tracker.close(round(elapsed, 3))
            summary['segments'] = tracker.finish(samples)
        if sampler is not None and sampler.fell_back_to_device:
            # Fallback ticks can carry other processes' memory, so the
            # own-only peaks are reported alongside rather than silently mixed.
            summary['per_gpu_excluding_fallback'] = _per_gpu_stats(
                [smp for smp in samples if not smp[2]])

        # BOTH files are written unconditionally -- a failed or interrupted run
        # is exactly when the measurement is most wanted.  A write error here
        # must not mask the child's own exit code, so it is reported and
        # swallowed.
        written = []
        try:
            _write_csv(csv_path, samples)
            written.append(csv_path)
        except OSError as e:
            print('monitor_run: could not write %s (%s)' % (csv_path, e),
                  file=sys.stderr)
        try:
            with open(json_path, 'w') as fp:
                json.dump(summary, fp, indent=2)
                fp.write('\n')
            written.append(json_path)
        except OSError as e:
            print('monitor_run: could not write %s (%s)' % (json_path, e),
                  file=sys.stderr)

        # --- the 3-line summary, on stdout ---
        print('wall time: %.2f s (%s), exit %d'
              % (elapsed, _hms(elapsed), exit_code))
        if sampler is None:
            print('peak GPU memory: not measured (--gpu-mem not given)')
        elif sampler.disabled_reason:
            print('peak GPU memory: not measured (%s)'
                  % sampler.disabled_reason)
        elif not stats:
            print('peak GPU memory: no samples (target finished inside one '
                  '%.3gs interval)' % a.interval)
        else:
            note = ' [some ticks are whole-device readings]' \
                if sampler.fell_back_to_device else ''
            print('peak GPU memory (%s): %s%s'
                  % (a.gpu_mem_scope,
                     ', '.join('gpu%s %.0f MiB' % (k, v['peak_mib'])
                               for k, v in stats.items()), note))
        if tracker is not None and summary.get('segments'):
            for seg in summary['segments']:
                dur = ('%.2f s' % seg['duration_seconds']
                       if seg['duration_seconds'] is not None else '?')
                own = seg['peak_mib_excluding_fallback']
                use, tag = ((own, '') if own
                            else (seg['peak_mib'], ' [whole-device]'))
                pk = (', '.join('gpu%s %.0f MiB' % (k, v)
                                for k, v in use.items()) + tag
                      or 'no samples')
                print('  segment %-14s %8s  %s'
                      % (seg['subcommand'], dur, pk))
        print('reports: %s' % (', '.join(written) if written
                               else '(none written)'))
        sys.stdout.flush()

    return exit_code


def _hms(seconds):
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return '%dh%02dm%02ds' % (h, m, s)
    if m:
        return '%dm%02ds' % (m, s)
    return '%ds' % s


if __name__ == '__main__':
    sys.exit(main())
