#!/usr/bin/env python
# coding: utf-8

"""
Advanced script to manage data for Speaker Verification project in Colab runtime.
Version 3: Uses --dataset_keys to specify which dataset configs to process.

Supports selective actions via flags:
- Copy original datasets (raw/spec) from Drive to Local.
- Pre-process local raw audio (.wav) into fixed-length numpy arrays (.npy),
  saving them locally AND optionally to Drive.
- Copy existing pre-processed .npy files from Drive to Local.
- Generate pair CSV files pointing to either local or Drive .npy files.
"""

import argparse
import csv
import os
import shutil
import time
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count, Manager

# Suppress specific librosa warnings if desired
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')

import librosa
import numpy as np
import soundfile as sf
import torch # Only needed for set_start_method if __main__
from tqdm import tqdm

# --- Constants ---
TARGET_SR_DEFAULT = 16000
TARGET_LEN_DEFAULT = 48000
LOCAL_BASE_DIR_DEFAULT = "/content/data"
DRIVE_NPY_DIR_DEFAULT_SUFFIX = "data/raw_audio_preprocessed/" # Relative to workspace

# --- Helper Functions (Mostly unchanged) ---
# (find_unique_files_from_csv, copy_file, copy_files_parallel,
#  preprocess_single_raw_file, preprocess_raw_files_parallel,
#  generate_npy_csv - remain the same as in data_manager_v2_script_fix2)
# ... (Include the full helper functions from data_manager_v2_script_fix2 here) ...
def find_unique_files_from_csv(csv_path, drive_workspace):
    """Reads a pair CSV (paths relative to workspace) and returns a set of unique, absolute file paths on Drive."""
    unique_files = set()
    if not csv_path.exists():
        print(f"Warning: CSV file not found at {csv_path}, cannot extract files.")
        return unique_files
    try:
        with open(csv_path, 'r', newline='') as infile:
            reader = csv.reader(infile)
            header = next(reader)
            for row in reader:
                if len(row) == 3:
                    f1_rel, f2_rel, _ = row
                    f1_rel = f1_rel.strip(); f2_rel = f2_rel.strip()
                    if not f1_rel or not f2_rel: continue
                    abs_path1 = (drive_workspace / f1_rel).resolve()
                    abs_path2 = (drive_workspace / f2_rel).resolve()
                    unique_files.add(abs_path1)
                    unique_files.add(abs_path2)
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
    return unique_files

def copy_file(args_tuple):
    """Copies a single file. Returns status string."""
    src_path, dest_path, force_copy, shared_dict = args_tuple
    status = 'error'
    try:
        if not dest_path.exists() or force_copy:
            if not src_path.exists():
                shared_dict['copy_errors_src_missing'] += 1
                status = 'error_src_missing'
            else:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dest_path)
                shared_dict['copied'] += 1
                status = 'copied'
        else:
            shared_dict['skipped'] += 1
            status = 'skipped'
    except Exception as e:
        shared_dict['copy_errors_other'] += 1
        status = 'error'
    return status

def copy_files_parallel(file_set, drive_source_base, local_dest_base, force_copy, desc="Copying files"):
    """Copies files in parallel. Returns set of LOCAL destination paths."""
    tasks = []
    local_paths_map = {}
    missing_source_files = 0
    path_errors = 0
    print(f"Preparing copy tasks for {desc}...")
    for src_path in file_set:
        try:
            if not src_path.exists():
                missing_source_files += 1
                continue
            if not drive_source_base.is_absolute():
                 print(f"ERROR: drive_source_base '{drive_source_base}' must be absolute.")
                 path_errors += 1
                 continue
            relative_path = src_path.relative_to(drive_source_base)
            dest_path = local_dest_base / relative_path
            tasks.append((src_path, dest_path))
            local_paths_map[src_path] = dest_path
        except ValueError: path_errors += 1
        except Exception as e: print(f"Error preparing copy task for {src_path}: {e}"); path_errors += 1

    if missing_source_files > 0: print(f"Warning: {missing_source_files} source files not found on Drive.")
    if path_errors > 0: print(f"Warning: {path_errors} files skipped due to path issues.")
    if not tasks: print("No valid files found to copy."); return set()

    print(f"Starting parallel copy of {len(tasks)} files...")
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

def preprocess_single_raw_file(args_tuple):
    """Preprocesses a single raw audio file. Returns status string."""
    (wav_path, local_npy_base, local_wav_base, drive_npy_base, force_preprocess,
     target_sr, target_len, shared_dict) = args_tuple
    status = 'error'
    try:
        relative_path = wav_path.relative_to(local_wav_base)
        local_npy_path = local_npy_base / relative_path.with_suffix(".npy")
        drive_npy_path = (drive_npy_base / relative_path.with_suffix(".npy")) if drive_npy_base else None

        local_exists = local_npy_path.exists()
        drive_exists = drive_npy_path.exists() if drive_npy_path else True

        if not force_preprocess and local_exists and drive_exists:
            shared_dict['skipped'] += 1
            return 'skipped'

        # Load audio
        try: audio, _ = librosa.load(wav_path, sr=target_sr, mono=True)
        except Exception:
            try:
                audio_sf, sr_sf = sf.read(wav_path, dtype='float32')
                if sr_sf != target_sr: audio = librosa.resample(y=audio_sf, orig_sr=sr_sf, target_sr=target_sr)
                else: audio = audio_sf
                if audio.ndim > 1: audio = np.mean(audio, axis=1)
            except Exception: shared_dict['errors'] += 1; return 'error_load'

        # Pad/truncate
        current_length = audio.shape[0]
        if current_length > target_len: audio = audio[:target_len]
        elif current_length < target_len: audio = np.pad(audio, (0, target_len - current_length), mode='constant')
        if audio.shape[0] != target_len: shared_dict['errors'] += 1; return 'error_length'
        processed_audio = audio.astype(np.float32)

        # Save Locally
        save_local_ok = False
        try:
            if force_preprocess or not local_exists:
                local_npy_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(local_npy_path, processed_audio)
            save_local_ok = True
        except Exception as e: print(f"\nError saving LOCAL {local_npy_path}: {e}"); shared_dict['errors'] += 1

        # Save to Drive
        save_drive_ok = False
        if drive_npy_path:
            try:
                if force_preprocess or not drive_exists:
                    drive_npy_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(drive_npy_path, processed_audio)
                save_drive_ok = True
            except Exception as e: print(f"\nError saving DRIVE {drive_npy_path}: {e}"); shared_dict['errors'] += 1
        else: save_drive_ok = True

        if save_local_ok and save_drive_ok: status = 'processed'
        else: status = 'error_save'

    except Exception: status = 'error'
    if status.startswith('error'): shared_dict['errors'] += 1
    elif status == 'processed': shared_dict['processed'] += 1
    return status

def preprocess_raw_files_parallel(local_wav_paths_set, local_wav_base, local_npy_base, drive_npy_base,
                                  force_preprocess, target_sr, target_len, desc="Preprocessing"):
    """Preprocesses raw audio files in parallel."""
    tasks = [(p, local_npy_base, local_wav_base, drive_npy_base, force_preprocess, target_sr, target_len)
             for p in local_wav_paths_set if p.is_file()]
    if not tasks: print(f"No valid local raw files found for {desc}."); return

    print(f"Starting parallel preprocessing of {len(tasks)} raw files for {desc}...")
    num_processes = min(max(1, cpu_count() // 2), len(tasks))
    print(f"Using {num_processes} processes.")

    with Manager() as manager:
        shared_dict = manager.dict({'processed': 0, 'skipped': 0, 'errors': 0})
        tasks_with_dict = [t + (shared_dict,) for t in tasks]
        with Pool(processes=num_processes) as pool:
            with tqdm(total=len(tasks), desc=desc) as pbar:
                for _ in pool.imap_unordered(preprocess_single_raw_file, tasks_with_dict):
                    pbar.update(1)
        processed = shared_dict['processed']
        skipped = shared_dict['skipped']
        errors = shared_dict['errors']

    print(f"\nPreprocessing finished for {desc}: Processed: {processed}, Skipped: {skipped}, Errors: {errors}")

def generate_npy_csv(original_wav_csv_path, output_npy_csv_path, wav_base_dir, npy_base_dir, check_npy_exists, desc="Generating CSV"):
    """Generates a CSV pointing to NPY files based on an original WAV CSV."""
    print(f"Generating {desc} CSV: {output_npy_csv_path}")
    print(f"  Reading original pairs from: {original_wav_csv_path}")
    print(f"  Mapping .wav relative to: {wav_base_dir}")
    print(f"  Creating .npy paths relative to: {npy_base_dir}")

    written_count = 0
    skipped_count = 0
    if not original_wav_csv_path.exists():
        print(f"ERROR: Original WAV CSV not found at {original_wav_csv_path}")
        return False
    try:
        output_npy_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(original_wav_csv_path, 'r', newline='') as infile, \
             open(output_npy_csv_path, 'w', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            header = next(reader)
            writer.writerow(header)
            for row in tqdm(reader, desc=desc):
                if len(row) == 3:
                    wav_path1_str, wav_path2_str, label = row
                    try:
                        wav_path1 = Path(wav_path1_str)
                        wav_path2 = Path(wav_path2_str)
                        relative_path1 = wav_path1.relative_to(wav_base_dir)
                        relative_path2 = wav_path2.relative_to(wav_base_dir)
                        npy_path1 = (npy_base_dir / relative_path1).with_suffix(".npy")
                        npy_path2 = (npy_base_dir / relative_path2).with_suffix(".npy")

                        if check_npy_exists and (not npy_path1.exists() or not npy_path2.exists()):
                            skipped_count += 1
                            continue

                        writer.writerow([str(npy_path1), str(npy_path2), label])
                        written_count += 1
                    except ValueError as ve: skipped_count += 1
                    except Exception as e: skipped_count += 1; print(f"Error processing row {row}: {e}")
                else: skipped_count += 1

        print(f"Finished {desc}: Wrote {written_count} pairs. Skipped {skipped_count} pairs.")
        return True
    except Exception as e:
        print(f"ERROR during CSV generation for {output_npy_csv_path}: {e}")
        return False


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Manage data preparation for SV project.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Paths
    parser.add_argument("--drive_workspace_dir", type=Path, required=True, help="Path to the main project folder on Google Drive.")
    parser.add_argument("--local_base_dir", type=Path, default=Path(LOCAL_BASE_DIR_DEFAULT), help="Base path for local data storage in Colab.")
    parser.add_argument("--drive_npy_base_dir", type=Path, default=None, help=f"Base directory on Google Drive to save/load .npy files (defaults to '<drive_workspace_dir>/{DRIVE_NPY_DIR_DEFAULT_SUFFIX}').")
    # Preprocessing Params
    parser.add_argument("--target_sr", type=int, default=TARGET_SR_DEFAULT, help="Target sample rate for raw audio preprocessing.")
    parser.add_argument("--target_len", type=int, default=TARGET_LEN_DEFAULT, help="Target length (samples) for raw audio preprocessing.")
    # Action Flags
    parser.add_argument("--copy_raw", action="store_true", help="Copy original raw WAVs from Drive to Local.")
    parser.add_argument("--copy_spec", action="store_true", help="Copy spectrogram JPGs/PNGs from Drive to Local.")
    parser.add_argument("--preprocess_raw", action="store_true", help="Generate NPY from local WAVs (saves local+Drive).")
    parser.add_argument("--load_npy", action="store_true", help="Copy existing NPYs from Drive to Local.")
    parser.add_argument("--gen_csv_local", action="store_true", help="Generate NPY CSVs pointing to LOCAL paths.")
    parser.add_argument("--gen_csv_drive", action="store_true", help="Generate NPY CSVs pointing to DRIVE paths.")
    # NEW: Specify which dataset keys to apply actions to
    parser.add_argument("--dataset_keys", nargs='+', required=True,
                        choices=['train_clean', 'train_noisy', 'test_clean', 'test_noisy', 'test_filtered'],
                        help="Specify one or more dataset keys to apply the actions to.")
    # Force Flags
    parser.add_argument("--force_copy", action="store_true", help="Overwrite existing files during copy actions.")
    parser.add_argument("--force_preprocess", action="store_true", help="Overwrite existing .npy files during preprocess actions.")

    args = parser.parse_args()
    start_script_time = time.time()

    # --- Path Definitions ---
    drive_ws = args.drive_workspace_dir
    local_base = args.local_base_dir
    drive_npy_base = args.drive_npy_base_dir or (drive_ws / DRIVE_NPY_DIR_DEFAULT_SUFFIX)

    # Define base paths for different dataset types
    # Keys match --dataset_keys choices
    configs = {
        "train_clean": {
            "drive_raw_base": Path("/content/drive/Shareddrives/VoxCeleb1/Dev/wav"), # Original clean WAVs
            "drive_spec_base": drive_ws / "data/spectrograms_generated_png/clean_train/", # Where clean PNGs are stored
            "local_raw_base": local_base / "raw_audio/clean_train/",
            "local_spec_base": local_base / "spectrograms/clean_train/",
            "local_npy_base": local_base / "raw_audio_preprocessed/clean_train/",
            "drive_npy_base": drive_npy_base / "clean_train/",
            "orig_raw_csv": drive_ws / "data/pairs/pairs_raw_clean_train.csv", # CSV listing clean WAV pairs
            "orig_spec_csv": drive_ws / "data/pairs/pairs_spec_clean_train_drive.csv", # CSV listing clean spec pairs (Drive paths)
            "out_npy_local_csv": drive_ws / "data/pairs/pairs_raw_clean_train_preprocessed_local.csv",
            "out_npy_drive_csv": drive_ws / "data/pairs/pairs_raw_clean_train_preprocessed_drive.csv",
        },
        "train_noisy": {
            "drive_raw_base": Path("/content/drive/Shareddrives/VoxCeleb1/Dev_Augmented/wav"), # Noisy WAVs
            "drive_spec_base": drive_ws / "data/spectrograms_generated_png/noisy_train/", # Where noisy PNGs are stored
            "local_raw_base": local_base / "raw_audio/noisy_train/",
            "local_spec_base": local_base / "spectrograms/noisy_train/",
            "local_npy_base": local_base / "raw_audio_preprocessed/noisy_train/",
            "drive_npy_base": drive_npy_base / "noisy_train/",
            "orig_raw_csv": drive_ws / "data/pairs/pairs_raw_train.csv", # CSV listing noisy WAV pairs
            "orig_spec_csv": drive_ws / "data/pairs/pairs_spec_train_drive.csv", # CSV listing noisy spec pairs (Drive paths)
            "out_npy_local_csv": drive_ws / "data/pairs/pairs_raw_train_preprocessed_local.csv",
            "out_npy_drive_csv": drive_ws / "data/pairs/pairs_raw_train_preprocessed_drive.csv",
        },
        "test_clean": {
            "drive_raw_base": Path("/content/drive/Shareddrives/VoxCeleb1/Test/wav/wav"),
            "drive_spec_base": drive_ws / "data/spectrograms_generated_png/clean_test/",
            "local_raw_base": local_base / "raw_audio/clean_test/",
            "local_spec_base": local_base / "spectrograms/clean_test/",
            "local_npy_base": local_base / "raw_audio_preprocessed/clean_test/",
            "drive_npy_base": drive_npy_base / "clean_test/",
            "orig_raw_csv": drive_ws / "data/pairs/pairs_raw_clean_test.csv",
            "orig_spec_csv": drive_ws / "data/pairs/pairs_spec_clean_test_drive.csv",
            "out_npy_local_csv": drive_ws / "data/pairs/pairs_raw_clean_test_preprocessed_local.csv",
            "out_npy_drive_csv": drive_ws / "data/pairs/pairs_raw_clean_test_preprocessed_drive.csv",
        },
         "test_noisy": {
            "drive_raw_base": Path("/content/drive/Shareddrives/VoxCeleb1/Test_Augmented/wav/wav"),
            "drive_spec_base": drive_ws / "data/spectrograms_generated_png/noisy_test/",
            "local_raw_base": local_base / "raw_audio/noisy_test/",
            "local_spec_base": local_base / "spectrograms/noisy_test/",
            "local_npy_base": local_base / "raw_audio_preprocessed/noisy_test/",
            "drive_npy_base": drive_npy_base / "noisy_test/",
            "orig_raw_csv": drive_ws / "data/pairs/pairs_raw_noisy_test.csv",
            "orig_spec_csv": drive_ws / "data/pairs/pairs_spec_noisy_test_drive.csv",
            "out_npy_local_csv": drive_ws / "data/pairs/pairs_raw_noisy_test_preprocessed_local.csv",
            "out_npy_drive_csv": drive_ws / "data/pairs/pairs_raw_noisy_test_preprocessed_drive.csv",
        },
         "test_filtered": {
            "drive_raw_base": Path("/content/drive/Shareddrives/VoxCeleb1/Test_Filtered/wav"),
            "drive_spec_base": drive_ws / "data/spectrograms_generated_png/filtered_test/",
            "local_raw_base": local_base / "raw_audio/filtered_test/",
            "local_spec_base": local_base / "spectrograms/filtered_test/",
            "local_npy_base": local_base / "raw_audio_preprocessed/filtered_test/",
            "drive_npy_base": drive_npy_base / "filtered_test/",
            "orig_raw_csv": drive_ws / "data/pairs/pairs_raw_filtered_test.csv",
            "orig_spec_csv": drive_ws / "data/pairs/pairs_spec_filtered_test_drive.csv",
            "out_npy_local_csv": drive_ws / "data/pairs/pairs_raw_filtered_test_preprocessed_local.csv",
            "out_npy_drive_csv": drive_ws / "data/pairs/pairs_raw_filtered_test_preprocessed_drive.csv",
        },
        # Add more configurations if needed
    }

    # --- Execute Actions ---
    active_flags = {action for action, value in vars(args).items()
                    if action.startswith(('copy_', 'preprocess_', 'load_', 'gen_')) and value}
    print(f"\nApplying actions {active_flags} to datasets: {', '.join(args.dataset_keys)}")

    # Store local raw paths resulting from copy action, used by preprocess
    local_copied_raw_paths = {key: set() for key in configs}

    for key in args.dataset_keys:
        if key not in configs:
            print(f"Warning: Unknown dataset key '{key}'. Skipping.")
            continue
        print(f"\n--- Processing Dataset: {key} ---")
        cfg = configs[key]

        # --- Copy Actions ---
        if args.copy_raw:
            print("Action: Copying Raw WAVs...")
            files = find_unique_files_from_csv(cfg["orig_raw_csv"], drive_ws)
            copied_paths = copy_files_parallel(files, cfg["drive_raw_base"], cfg["local_raw_base"], args.force_copy, desc=f"Copying Raw ({key})")
            local_copied_raw_paths[key].update(copied_paths)
        if args.copy_spec:
            print("Action: Copying Spectrograms...")
            files = find_unique_files_from_csv(cfg["orig_spec_csv"], drive_ws) # Use spec CSV to find spec files
            copy_files_parallel(files, cfg["drive_spec_base"], cfg["local_spec_base"], args.force_copy, desc=f"Copying Spec ({key})")

        # --- Preprocess Action ---
        if args.preprocess_raw:
            print("Action: Preprocessing Raw WAVs to NPY...")
            paths_to_process = local_copied_raw_paths[key]
            if not paths_to_process and not args.copy_raw: # If copy wasn't run this time
                 print(f"Scanning local dir {cfg['local_raw_base']} for WAV files...")
                 paths_to_process = set(cfg["local_raw_base"].rglob('*.wav'))
                 print(f"Found {len(paths_to_process)} files.")
            if paths_to_process:
                 preprocess_raw_files_parallel(paths_to_process, cfg["local_raw_base"], cfg["local_npy_base"],
                                               cfg["drive_npy_base"], args.force_preprocess,
                                               args.target_sr, args.target_len, desc=f"Preprocessing Raw ({key})")
            else: print("No local WAV files found. Skipping preprocessing.")

        # --- Load NPY Action ---
        if args.load_npy:
            print("Action: Loading NPYs from Drive...")
            expected_npy_files_on_drive = set()
            # Use the *original raw* CSV to determine which NPYs correspond
            wav_files = find_unique_files_from_csv(cfg["orig_raw_csv"], drive_ws)
            for wav_path in wav_files:
                 try:
                     rel_path = wav_path.relative_to(cfg["drive_raw_base"])
                     expected_npy_files_on_drive.add((cfg["drive_npy_base"] / rel_path).with_suffix(".npy"))
                 except ValueError: pass
            copy_files_parallel(expected_npy_files_on_drive, cfg["drive_npy_base"], cfg["local_npy_base"], args.force_copy, desc=f"Loading NPY ({key})")

        # --- Generate CSV Actions ---
        if args.gen_csv_local:
            print("Action: Generating Local NPY CSV...")
            generate_npy_csv(cfg["orig_raw_csv"], cfg["out_npy_local_csv"],
                             cfg["drive_raw_base"], cfg["local_npy_base"], True, # Check local NPY exists
                             desc=f"Gen Local NPY CSV ({key})")
        if args.gen_csv_drive:
            print("Action: Generating Drive NPY CSV...")
            generate_npy_csv(cfg["orig_raw_csv"], cfg["out_npy_drive_csv"],
                             cfg["drive_raw_base"], cfg["drive_npy_base"], False, # Don't check Drive NPY exists
                             desc=f"Gen Drive NPY CSV ({key})")


    print(f"\nScript finished. Total time: {time.time() - start_script_time:.2f} seconds.")

if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass
    main()
