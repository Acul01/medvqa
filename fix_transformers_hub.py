#!/usr/bin/env python3
"""
Root-cause fix for the hub.py bug that rejects local absolute paths.

cached_files() calls hf_hub_download(repo_id=local_path) even for local
directories when a file is absent — hf_hub_download's validator then rejects
the path format with HFValidationError.  All callers that pass
_raise_exceptions_for_missing_entries=False (like maybe_load_adapters) expect
None to be returned for missing files, not a crash.

Fix: after the "all-files-found" early return, add an early return for local
directories so they never fall through to hf_hub_download.

Run once:  python fix_transformers_hub.py
Revert:    cp <file>.original_bak <file>
"""
import inspect
import shutil
import sys

import transformers.utils.hub as hub_module

filepath = inspect.getfile(hub_module)
print(f"Target : {filepath}")

backup = filepath + ".original_bak"
shutil.copy2(filepath, backup)
print(f"Backup : {backup}")

with open(filepath) as fh:
    src = fh.read()

if "# PATCHED_HUB_LOCAL_FIX" in src:
    print("Already patched — nothing to do.")
    sys.exit(0)

# Exact string to find (from `sed -n '413,418p' hub.py`)
OLD = '''\
    # Either all the files were found, or some were _CACHED_NO_EXIST but we do not raise for missing entries
    if file_counter == len(full_filenames):
        return existing_files if len(existing_files) > 0 else None

    user_agent = http_user_agent(user_agent)
    # download the files if needed'''

NEW = '''\
    # Either all the files were found, or some were _CACHED_NO_EXIST but we do not raise for missing entries
    if file_counter == len(full_filenames):
        return existing_files if len(existing_files) > 0 else None

    # PATCHED_HUB_LOCAL_FIX: local directories must never fall through to Hub
    # download.  hf_hub_download validates its first argument as a Hub repo-id
    # and rejects local absolute paths with HFValidationError (→ OSError).
    # For local dirs with missing files, either raise EnvironmentError (caught
    # upstream as OSError) or return None/empty when errors are suppressed.
    if os.path.isdir(str(path_or_repo_id)):
        if _raise_exceptions_for_missing_entries:
            raise EnvironmentError(
                f"Could not locate some file(s) in local directory {str(path_or_repo_id)!r}."
            )
        return existing_files if len(existing_files) > 0 else None

    user_agent = http_user_agent(user_agent)
    # download the files if needed'''

if OLD not in src:
    print("ERROR: expected block not found in hub.py.")
    print("The transformers version may differ. Try:")
    print(f"  grep -n 'file_counter\\|user_agent\\|download the files' {filepath}")
    sys.exit(1)

new_src = src.replace(OLD, NEW, 1)

with open(filepath, "w") as fh:
    fh.write(new_src)

print("Patch applied.")
print()
print("Verify:")
print('  python -c "from transformers.utils.hub import cached_files; print(\'OK\')"')
print()
print("To revert:")
print(f"  cp {backup} {filepath}")
