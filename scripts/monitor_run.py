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
import json
import os
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
    ap.add_argument('--out-dir', default=None,
                    help='directory to write the measurement summary into; '
                         'created if missing.  Omit to print only.')
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
    try:
        proc = subprocess.Popen(target, start_new_session=True)
    except FileNotFoundError:
        print('monitor_run: cannot execute %r -- no such file' % target[0],
              file=sys.stderr)
        return 127
    except PermissionError:
        print('monitor_run: cannot execute %r -- not executable' % target[0],
              file=sys.stderr)
        return 126

    sampler = None
    if a.gpu_mem:
        sampler = GpuSampler(proc.pid, a.interval, a.gpu_mem_scope,
                             t0_mono).start()

    prev = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        prev[sig] = signal.signal(sig, _forward_to_group(proc.pid))
    try:
        rc = proc.wait()
    finally:
        for sig, old in prev.items():
            signal.signal(sig, old)
        if sampler is not None:
            sampler.stop()
    elapsed = time.monotonic() - t0_mono
    t1 = time.time()

    # A child killed by signal N reports -N; report it the way a shell does.
    exit_code = rc if rc >= 0 else 128 + (-rc)

    if a.runtime:
        print('monitor_run: wall clock %.2f s (%s)'
              % (elapsed, _hms(elapsed)), file=sys.stderr)
    if rc < 0:
        print('monitor_run: target killed by signal %d -> exit %d'
              % (-rc, exit_code), file=sys.stderr)
    else:
        print('monitor_run: target exited %d' % exit_code, file=sys.stderr)
    if a.gpu_mem:
        if sampler.disabled_reason:
            print('monitor_run: no GPU memory measured (%s)'
                  % sampler.disabled_reason, file=sys.stderr)
        elif not sampler.samples:
            print('monitor_run: GPU sampling ran but collected no samples '
                  '(target finished inside one %.3gs interval)' % a.interval,
                  file=sys.stderr)
        else:
            peak = sampler.peaks()
            scope = ('device' if a.gpu_mem_scope == 'device'
                     else ('proc-tree, fell back to device on some ticks'
                           if sampler.fell_back_to_device else 'proc-tree'))
            print('monitor_run: peak GPU memory (%s, %d samples): %s'
                  % (scope, len(sampler.samples),
                     ', '.join('gpu%d %.0f MiB' % (i, peak[i])
                               for i in sorted(peak))), file=sys.stderr)

    summary = dict(
        command=target,
        wall_seconds=round(elapsed, 3),
        wall_hms=_hms(elapsed),
        started_unix=round(t0, 3),
        ended_unix=round(t1, 3),
        exit_code=exit_code,
        killed_by_signal=(-rc if rc < 0 else None),
        measured=dict(runtime=bool(a.runtime), gpu_mem=bool(a.gpu_mem)),
        interval=a.interval,
        gpu_mem_scope=a.gpu_mem_scope,
        gpu_mem=None,
        gpu_mem_note=None,
    )
    if a.gpu_mem:
        summary['gpu_mem'] = dict(
            scope=a.gpu_mem_scope,
            fell_back_to_device=bool(sampler.fell_back_to_device),
            n_samples=len(sampler.samples),
            peak_mib={str(i): v for i, v in sorted(sampler.peaks().items())},
        )
        summary['gpu_mem_note'] = sampler.disabled_reason
        summary['gpu_mem']['peak_mib_excluding_fallback'] = {
            str(i): v for i, v in sorted(sampler.peaks(own_only=True).items())}
        summary['gpu_samples'] = [
            dict(t_seconds=t,
                 used_mib={str(i): v for i, v in sorted(g.items())},
                 device_fallback=bool(fb))
            for t, g, fb in sampler.samples]
    if out_dir:
        path = os.path.join(out_dir, 'monitor_run.json')
        with open(path, 'w') as fp:
            json.dump(summary, fp, indent=2)
            fp.write('\n')
        print('monitor_run: wrote %s' % path, file=sys.stderr)

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
