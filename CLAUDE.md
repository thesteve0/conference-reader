# Claude Code Instructions

## Command Output Requirements

**FORBIDDEN**: Using `| tail`, `| less`, `| head`, or similar filters to truncate command output.

**REQUIRED**: Capture all log output from commands in their entirety.

**ALLOWED**: Using `grep` to search for specific words or phrases within output.

This instruction MUST BE FOLLOWED in every session.

## Package Installation

**REQUIRED**: All Python packages must be installed using `uv add`, not `pip install`.

Example: `uv add pytesseract` (correct), NOT `pip install pytesseract` (incorrect).
