#!/usr/bin/env python
# coding: utf-8

"""
Copies pre-processed .npy files listed in a CSV from Google Drive
to the local Colab runtime storage.
"""

import argparse
import csv
import os
import shutil
import torch
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm

# --- Helper Functions ---

def copy_file(args_tuple):
    """Copies a single file. Returns status string."""
    src_path, dest_path, force_copy, shared_dict = args_tuple
    status = 'error'
    try:
        # Ensure destination parent directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if not dest_path.exists() or force_copy:
            if not src_path.exists():
                shared_dict['copy_errors_src_missing'] += 1
                status = 'error_src_missing'
                # print(f"Source missing: {src_path}") # Optional debug
            else:
                shutil.copy2(src_path, dest_path)
                shared_dict['copied'] += 1
                status = 'copied'
        else:
            shared_dict['skipped'] += 1
            status = 'skipped'
    except Exception as e:
        # print(f"\nError copying {src_path} to {dest_path}: {e}") # Optional debug
        shared_dict['copy_errors_other'] += 1
        status = 'error'
    return status

def copy_files_parallel(file_set, local_dest_base, force_copy, desc="Copying NPY files"):
    """Copies files in parallel. Expects file_set to contain absolute Drive paths."""
    tasks = []
    local_paths_map = {}
    missing_source_files = 0
    path_errors = 0
    print(f"Preparing copy tasks for {desc}...")

    # Determine a common base path on Drive from the source files to calculate relative paths
    # This assumes the .npy files on drive share a common structure root.
    # Example: If files are /drive/.../base/noisy/f1.npy, /drive/.../base/noisy/f2.npy
    # common_drive_base might be /drive/.../base/
    # If they don't share a common base, relative path calculation will fail.
    drive_paths_list = [str(p) for p in file_set]
    if not drive_paths_list:
        print("No source files provided.")
        return set()

    try:
        # Find the longest common directory path among all source files
        common_drive_base = Path(os.path.commonpath(drive_paths_list)).parent # Go one level up from common dir usually
        print(f"Inferred common Drive base for relative paths: {common_drive_base}")
    except ValueError:
        print("Warning: Could not determine a common base path for source NPY files on Drive.")
        print("Cannot calculate relative paths reliably. Ensure target local directory structure exists.")
        # Fallback: Use full path hashing or another method if needed, but simple copy might fail structure.
        # For now, we'll proceed assuming relative paths might work or structure isn't critical.
        common_drive_base = None # Indicate failure to find common base


    for src_path in file_set:
        try:
            if not src_path.exists():
                missing_source_files += 1
                continue

            if common_drive_base:
                 # Calculate path relative to the inferred common base
                 relative_path = src_path.relative_to(common_drive_base)
                 # Construct destination path preserving structure
                 dest_path = (local_dest_base / relative_path).resolve()
            else:
                 # Fallback: Just put the file directly in the base dest dir (loses structure)
                 print(f"Warning: Using fallback destination path for {src_path.name}")
                 dest_path = (local_dest_base / src_path.name).resolve()


            tasks.append((src_path, dest_path))
            local_paths_map[src_path] = dest_path
        except ValueError:
            # This might happen if a file isn't relative to the common_drive_base
            print(f"Warning: Path {src_path} not relative to inferred base {common_drive_base}. Skipping.")
            path_errors += 1
        except Exception as e:
             print(f"Error preparing copy task for {src_path}: {e}")
             path_errors += 1

    if missing_source_files > 0: print(f"Warning: {missing_source_files} source NPY files not found on Drive.")
    if path_errors > 0: print(f"Warning: {path_errors} NPY files skipped due to path issues.")
    if not tasks: print("No valid NPY files found to copy."); return set()

    print(f"Starting parallel copy of {len(tasks)} NPY files...")
    num_processes = min(max(1, cpu_count() // 2), len(tasks))
    print(f"Using {num_processes} processes.")

    copied_source_paths = set()
    with Manager() as manager:
        shared_dict = manager.dict({'copied': 0, 'skipped': 0, 'copy_errors_src_missing': 0, 'copy_errors_other': 0})
        tasks_with_dict = [(s, d, force_copy, shared_dict) for s, d in tasks]
        with Pool(processes=num_processes) as pool:
            with tqdm(total=len(tasks), desc=desc) as pbar:
                for i, result_status in enumerate(pool.imap_unordered(copy_file, tasks_with_dict)):
                    original_src = tasks[i][0]
                    if result_status in ['copied', 'skipped']:
                        copied_source_paths.add(original_src)
                    pbar.update(1)
        copied_count = shared_dict['copied']
        skipped_count = shared_dict['skipped']
        error_count_missing = shared_dict['copy_errors_src_missing']
        error_count_other = shared_dict['copy_errors_other']

    print(f"\nCopy finished for {desc}: Copied: {copied_count}, Skipped: {skipped_count}, Src Missing Errors: {error_count_missing}, Other Errors: {error_count_other}")
    local_copied_paths = {local_paths_map[src] for src in copied_source_paths if src in local_paths_map}
    return local_copied_paths


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Copy pre-processed .npy files listed in a CSV from Drive to Local.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Required Paths
    parser.add_argument("--npy_csv_drive", type=Path, required=True,
                        help="Path to the CSV file ON DRIVE containing pairs of NPY file paths (absolute or relative to workspace).")
    parser.add_argument("--drive_workspace_dir", type=Path, required=True,
                        help="Path to the main project folder on Google Drive (used to resolve relative paths in CSV).")
    parser.add_argument("--local_npy_base_dir", type=Path, required=True,
                        help="Base path for storing copied .npy files locally in Colab (e.g., /content/data/raw_audio_preprocessed/noisy_train/).")
    # Optional Flags
    parser.add_argument("--force_copy", action="store_true",
                        help="Overwrite existing local .npy files during copy action.")

    args = parser.parse_args()
    start_script_time = time.time()

    # --- Validate Paths ---
    if not args.drive_workspace_dir.exists():
        print(f"ERROR: Drive workspace directory not found: {args.drive_workspace_dir}")
        return
    if not args.npy_csv_drive.exists():
        print(f"ERROR: NPY CSV file not found on Drive: {args.npy_csv_drive}")
        return

    # Ensure local destination base directory exists
    args.local_npy_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Local destination base directory: {args.local_npy_base_dir}")

    # --- Read NPY paths from CSV ---
    print(f"Reading NPY file paths from: {args.npy_csv_drive}")
    unique_npy_drive_paths = set()
    try:
        with open(args.npy_csv_drive, 'r', newline='') as infile:
            reader = csv.reader(infile)
            header = next(reader) # Skip header
            for i, row in enumerate(reader):
                if len(row) == 3:
                    npy_path1_str, npy_path2_str, _ = row
                    npy_path1_str = npy_path1_str.strip()
                    npy_path2_str = npy_path2_str.strip()
                    if not npy_path1_str or not npy_path2_str: continue # Skip empty paths

                    # Resolve paths relative to workspace if they are not absolute
                    path1 = Path(npy_path1_str)
                    path2 = Path(npy_path2_str)

                    abs_path1 = (args.drive_workspace_dir / path1).resolve() if not path1.is_absolute() else path1.resolve()
                    abs_path2 = (args.drive_workspace_dir / path2).resolve() if not path2.is_absolute() else path2.resolve()

                    unique_npy_drive_paths.add(abs_path1)
                    unique_npy_drive_paths.add(abs_path2)
                # else: print(f"Skipping malformed row {i+1}: {row}") # Optional
    except Exception as e:
        print(f"ERROR reading NPY CSV file {args.npy_csv_drive}: {e}")
        return

    print(f"Found {len(unique_npy_drive_paths)} unique NPY file paths to copy from Drive.")

    if not unique_npy_drive_paths:
        print("No NPY files found in CSV to copy.")
    else:
        # --- Copy NPY files ---
        copy_files_parallel(unique_npy_drive_paths, args.local_npy_base_dir, args.force_copy,
                            desc=f"Copying NPY to {args.local_npy_base_dir.name}")

    print(f"\nScript finished. Total time: {time.time() - start_script_time:.2f} seconds.")

if __name__ == "__main__":
    # Set start method for multiprocessing (recommended for CUDA compatibility)
    try:
        # Use 'fork' if possible on Linux for efficiency, fallback to 'spawn'
        torch.multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        try:
             torch.multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
             print("Note: Multiprocessing start method already set or cannot be forced.")
             pass
    main()
