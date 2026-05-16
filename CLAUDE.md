# Claude Code instructions for pochoir

## Task and planning workflow

Always use beads for planning and task tracking. Before writing any code, create a beads issue with `bd create`. Do NOT use TodoWrite, TaskCreate, or markdown files for task tracking.

**Planning strategy:** Invest heavily in planning before any implementation. A detailed, refined plan leads to straightforward execution with fewer bugs. The workflow is:

1. Create a beads issue immediately when a task is identified — do NOT wait for user confirmation.
2. Use plan mode to produce a thorough plan: explore the codebase, identify all affected files and functions, design the exact changes, and write explicit verification steps.
3. Only begin implementation once the plan is approved. Update the beads issue if the plan changes.
4. Close the beads issue after the code is committed.

## Git workflow

NEVER run `git push` under any circumstances. The user handles all pushes manually.

## Session discipline

Work on one phase at a time. After completing a phase (issue closed, code committed), ask the user to start a new Claude Code session before beginning the next phase. This keeps context focused and prevents carry-over state from polluting subsequent work.

## 🚨 Testing rules — ALWAYS APPLY 🚨

These rules govern every test/verification run. Re-read at the start of any session that involves running `pochoir`, FDM, or numerical verification.

1. **Use the project environment.** Always prepend `PATH=/nfs/data/1/rrazakami/work/pochoir/env/bin:$PATH` (or `source` its activate) for any `pochoir` CLI calls and Python helpers that depend on the project's deps. Do not use system Python for verification.
2. **Run long calculations in the background.** Long FDM/drift/induce pipelines must use `run_in_background=true` Bash calls. Do not block the conversation waiting for them. Only report when the run finishes or fails.
3. **Surface only results and blocking issues.** While a background run is in progress, do not narrate intermediate status to the user. When it finishes, report:
   - results of the verification checks (PASS/FAIL per check), and
   - any issues encountered that you had to solve.
4. **Self-resolve test issues.** If a test fails, a script needs fixing, or an environment problem surfaces during testing, fix it directly without asking the user. Only escalate if the fix would change the scope of the planned work or destroy data.
5. **End-of-run summary required.** After every successful test/verification, post a concise summary: what was run, what checks passed, files modified, beads issue closed. If the executed work diverged from the approved plan (e.g. extra fixes, scope changes due to bugs found), state the divergence explicitly in the summary.
