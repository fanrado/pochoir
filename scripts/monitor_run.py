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

    prev = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        prev[sig] = signal.signal(sig, _forward_to_group(proc.pid))
    try:
        rc = proc.wait()
    finally:
        for sig, old in prev.items():
            signal.signal(sig, old)
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
        print('monitor_run: --gpu-mem requested but GPU sampling is NOT '
              'implemented yet (Phase 1/Step 1 is the wrapper and wall clock); '
              'no memory figure was measured.', file=sys.stderr)

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
        gpu_mem_note=('sampler not implemented in Phase 1/Step 1'
                      if a.gpu_mem else None),
    )
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
