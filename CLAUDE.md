# Claude Code Instructions

## Command Output Requirements

**FORBIDDEN**: Using `| tail`, `| less`, `| head`, or similar filters to truncate command output.

**REQUIRED**: Capture all log output from commands in their entirety.

**ALLOWED**: Using `grep` to search for specific words or phrases within output.

This instruction MUST BE FOLLOWED in every session.

## Package Installation

**REQUIRED**: All Python packages must be installed using `uv add`, not `pip install`.

Example: `uv add pytesseract` (correct), NOT `pip install pytesseract` (incorrect).

## Git Commit Reminders

**REQUIRED**: After completing a working feature, fixing a bug, or reaching a logical stopping point, ask the user: "This seems like a good stopping point. Would you like me to create a git commit?"

Good stopping points include:
- A new feature works end-to-end
- A bug is fixed and verified
- A refactor is complete and tests pass
- Before starting a risky or experimental change
- After adding a new file or module

This helps ensure frequent commits so changes can be easily rolled back if needed.
