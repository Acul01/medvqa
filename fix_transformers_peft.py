#!/usr/bin/env python3
"""
One-time patch for a bug in transformers' maybe_load_adapters.

The function calls cached_file(local_abs_path, CONFIG_NAME, ...) which
internally calls hf_hub_download with the local path as repo_id.
huggingface_hub's validator rejects the path format and raises HFValidationError
(wrapped as OSError by cached_files), but maybe_load_adapters has no try/except.

Fix: wrap the cached_file call in try/except so local paths don't crash.

Run once:  python fix_transformers_peft.py
Revert:    cp <file>.original_bak <file>
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

# ── Read ──────────────────────────────────────────────────────────────────────
with open(filepath) as fh:
    src = fh.read()

if "# PATCHED_LOCAL_PATH_FIX" in src:
    print("Already patched — nothing to do.")
    sys.exit(0)

# ── Exact replacement ─────────────────────────────────────────────────────────
# OLD: the original block as it appears in the file (from the sed output).
# NEW: same block wrapped in try/except.
# Uses exact string matching — no indentation arithmetic needed.

OLD = '''\
    if download_kwargs.get("commit_hash") is None:
        resolved_config_file = cached_file(
            pretrained_model_name_or_path,
            CONFIG_NAME,
            cache_dir=download_kwargs.get("cache_dir"),
            force_download=bool(download_kwargs.get("force_download", False)),
            proxies=download_kwargs.get("proxies"),
            local_files_only=bool(download_kwargs.get("local_files_only", False)),
            token=token,
            revision=download_kwargs.get("revision"),
            subfolder=download_kwargs.get("subfolder"),
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )
        download_kwargs["commit_hash"] = extract_commit_hash(resolved_config_file, None)'''

NEW = '''\
    if download_kwargs.get("commit_hash") is None:
        try:  # PATCHED_LOCAL_PATH_FIX
            resolved_config_file = cached_file(
                pretrained_model_name_or_path,
                CONFIG_NAME,
                cache_dir=download_kwargs.get("cache_dir"),
                force_download=bool(download_kwargs.get("force_download", False)),
                proxies=download_kwargs.get("proxies"),
                local_files_only=bool(download_kwargs.get("local_files_only", False)),
                token=token,
                revision=download_kwargs.get("revision"),
                subfolder=download_kwargs.get("subfolder"),
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
        except (OSError, ValueError, Exception):
            resolved_config_file = None
        download_kwargs["commit_hash"] = extract_commit_hash(resolved_config_file, None)'''

if OLD not in src:
    print("ERROR: expected block not found in file.")
    print("The transformers version on this machine may differ.")
    print(f"Inspect manually: grep -n 'commit_hash\\|resolved_config_file' {filepath}")
    sys.exit(1)

new_src = src.replace(OLD, NEW, 1)

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
