#!/usr/bin/env python3
"""
One-time patch for a bug in transformers' maybe_load_adapters.

In some transformers versions, maybe_load_adapters passes local absolute paths
to hf_hub_download (which rejects them as invalid repo IDs) and has no
try-except to catch the resulting OSError.

Fix: wrap the cached_file("adapter_config.json") call in maybe_load_adapters
with try/except so that missing-file errors return None ("no adapter") instead
of crashing.

Run once on the server:
    python fix_transformers_peft.py
"""
import inspect
import shutil
import sys

import transformers.integrations.peft as peft_module

filepath = inspect.getfile(peft_module)
print(f"Target : {filepath}")

# ── Backup ────────────────────────────────────────────────────────────────────
backup = filepath + ".original_bak"
shutil.copy2(filepath, backup)
print(f"Backup : {backup}")

# ── Read source ───────────────────────────────────────────────────────────────
with open(filepath, "r") as fh:
    src = fh.read()

if "# PATCHED_LOCAL_PATH_FIX" in src:
    print("Already patched — nothing to do.")
    sys.exit(0)

# ── Locate the resolved_config_file = cached_file( call ──────────────────────
MARKER = "resolved_config_file = cached_file("
idx = src.find(MARKER)
if idx == -1:
    print(f"ERROR: cannot find {MARKER!r} in {filepath}")
    print("Inspect manually: grep -n 'resolved_config_file' " + filepath)
    sys.exit(1)

# Determine base indentation of that line
line_start = src.rfind("\n", 0, idx) + 1
indent = idx - line_start          # number of spaces before MARKER
ind = " " * indent

# Find end of the full call by matching parentheses
depth = 0
call_end = idx
for pos, ch in enumerate(src[idx:]):
    if ch == "(":
        depth += 1
    elif ch == ")":
        depth -= 1
        if depth == 0:
            call_end = idx + pos + 1
            break

# Include the rest of the line after the closing paren (e.g. a newline)
eol = src.find("\n", call_end)
if eol == -1:
    eol = len(src)
else:
    eol += 1  # include the newline itself

original_call = src[idx:eol]   # everything from MARKER to end of that line

# ── Build replacement ─────────────────────────────────────────────────────────
# The original_call slice starts directly at MARKER (no leading spaces), while
# every continuation line still carries its absolute indentation from the file.
#
# Required layout (example with ind=8, extra=4):
#   8sp  try:
#   12sp     resolved_config_file = cached_file(   ← first line: ind+extra
#   16sp         pretrained_model_name_or_path,    ← others:     extra+original
#   12sp     )
#   8sp  except ...:
#   12sp     resolved_config_file = None
#
# So: first line gets ind+extra, all continuation lines get only extra (they
# already have their original absolute indentation from the source file).
extra = "    "
call_lines = original_call.rstrip("\n").split("\n")
indented_lines = [ind + extra + call_lines[0]]          # first line: ind+extra
indented_lines += [extra + ln for ln in call_lines[1:]] # rest:  extra + absolute indent
indented = "\n".join(indented_lines)

replacement = (
    f"{ind}try:  # PATCHED_LOCAL_PATH_FIX\n"
    f"{indented}\n"
    f"{ind}except (OSError, ValueError, Exception):\n"
    f"{ind}{extra}resolved_config_file = None\n"
)

new_src = src[:idx] + replacement + src[eol:]

# ── Write ─────────────────────────────────────────────────────────────────────
with open(filepath, "w") as fh:
    fh.write(new_src)

print("Patch applied.")
print()
print("Verify:")
print('  python -c "import transformers.integrations.peft; print(\'OK\')"')
print()
print("To revert:")
print(f"  cp {backup} {filepath}")
