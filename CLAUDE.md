# Claude Code instructions for pochoir

## Task and planning workflow

Always use beads for planning and task tracking. Before writing any code, create a beads issue with `bd create`. Do NOT use TodoWrite, TaskCreate, or markdown files for task tracking.

**Planning strategy:** Invest heavily in planning before any implementation. A detailed, refined plan leads to straightforward execution with fewer bugs. The workflow is:

1. Create a beads issue immediately when a task is identified — do NOT wait for user confirmation.
2. Use plan mode to produce a thorough plan: explore the codebase, identify all affected files and functions, design the exact changes, and write explicit verification steps.
3. Only begin implementation once the plan is approved. Update the beads issue if the plan changes.
4. Close the beads issue after the code is committed.

## Session discipline

Work on one phase at a time. After completing a phase (issue closed, code committed), ask the user to start a new Claude Code session before beginning the next phase. This keeps context focused and prevents carry-over state from polluting subsequent work.
