# scripts/train_mobilenetv2.py (Optimized + AMP Loss + Compile/Resume Fix)

import argparse
import json
import os
import time
from pathlib import Path
import numpy as np # For isnan check
from collections import OrderedDict # For loading state dicts

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score
from torch.profiler import profile, record_function, ProfilerActivity # For Profiler

# Import local modules
try:
    from datasets.spectrogram_dataset import MelSpectrogramPairDataset
    from models.mobilenet_embedding import MobileNetV2Embedding
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Please ensure datasets/spectrogram_dataset.py and models/mobilenet_embedding.py exist.")
    exit(1)

# Check PyTorch version for torch.compile compatibility
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
COMPILE_SUPPORTED = (TORCH_MAJOR >= 2)
if not COMPILE_SUPPORTED:
    print(f"Warning: torch.compile requires PyTorch 2.0 or later. You have {torch.__version__}. Compilation will be skipped.")

# --- Pair Classifier (MODIFIED: Removed final sigmoid) ---
class PairClassifier(nn.Module):
    def __init__(self, emb_dim=256, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim*2, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.bn2 = nn.BatchNorm1d(hidden_dim//2)
        self.out = nn.Linear(hidden_dim//2,1) # Outputs raw logits
    def forward(self,e1,e2):
        x = torch.cat([e1,e2],dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        # REMOVED torch.sigmoid() here - output raw logits
        return self.out(x).squeeze(1)

# --- Evaluation Function (MODIFIED: Handles logits) ---
def evaluate(model, clf, loader, criterion, device, use_amp=True):
    model.eval(); clf.eval()
    total_loss = 0.0
    all_scores, all_labels = [], []
    amp_enabled = use_amp and (device.type == 'cuda')

    with torch.no_grad():
        for x1,x2,y in loader:
            x1,x2,y = x1.to(device), x2.to(device), y.float().to(device) # Labels to float

            autocast_args = {'device_type': device.type, 'enabled': amp_enabled}
            try:
                with torch.amp.autocast(**autocast_args):
                    e1 = model(x1); e2 = model(x2)
                    logits = clf(e1,e2) # Get logits
                    loss = criterion(logits,y) # Use logits
            except AttributeError:
                 with torch.cuda.amp.autocast(enabled=amp_enabled): # Fallback
                    e1 = model(x1); e2 = model(x2)
                    logits = clf(e1,e2)
                    loss = criterion(logits,y)

            total_loss += loss.item()*y.size(0)
            all_scores.append(logits.cpu())
            all_labels.append(y.cpu())

    if not hasattr(loader, 'dataset') or len(loader.dataset) == 0: avg_loss = float('nan')
    else: avg_loss = total_loss / len(loader.dataset)

    y_true = torch.cat(all_labels).numpy()
    y_score= torch.cat(all_scores).numpy() # Pass logits
    try: auc = roc_auc_score(y_true, y_score)
    except ValueError as e: print(f"Warning: Could not compute AUC in validation: {e}"); auc = 0.0
    return avg_loss, auc

# --- Path Remapping Function (Unchanged) ---
def remap_pairs(pairs, data_root):
    # (Same remapping logic as before)
    all_paths = [p for p,_,_ in pairs] + [q for _,q,_ in pairs];
    if not all_paths: print("Warning: No paths found in pairs list for remapping."); return []
    try: orig_root = os.path.commonpath(all_paths)
    except ValueError: print(f"Warning: Could not determine common path."); orig_root = None
    new = []; skipped_count = 0
    for p1,p2,l in pairs:
        try:
            if orig_root: rel1 = os.path.relpath(p1, orig_root); rel2 = os.path.relpath(p2, orig_root)
            else:
                 path1_obj = Path(p1); path2_obj = Path(p2)
                 if path1_obj.is_absolute() and str(path1_obj).startswith(str(data_root)): rel1 = path1_obj.relative_to(data_root)
                 else: rel1 = p1
                 if path2_obj.is_absolute() and str(path2_obj).startswith(str(data_root)): rel2 = path2_obj.relative_to(data_root)
                 else: rel2 = p2
            new_p1 = os.path.join(data_root, rel1); new_p2 = os.path.join(data_root, rel2)
            new.append((new_p1, new_p2, l))
        except ValueError as e: skipped_count += 1
        except Exception as e: skipped_count += 1; print(f"Error remapping pair ({p1}, {p2}): {e}")
    if skipped_count > 0: print(f"Skipped {skipped_count} pairs during path remapping.")
    return new

# --- Main Training Function ---
def main():
    p = argparse.ArgumentParser(description="Train MobileNetV2 Speaker Verification Model")
    # --- Arguments ---
    p.add_argument("csv", help="Path to the CSV file containing pairs of spectrogram image files (e.g., pairs_spec_train_local.csv)")
    p.add_argument("--data_root", type=str, default=None, help="Optional: Local root directory for spectrogram images.")
    p.add_argument("--epochs", type=int, default=50, help="Max number of training epochs.")
    p.add_argument("--batch_size", type=int, default=64, help="Training batch size.")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    p.add_argument("--val_split", type=float, default=0.1, help="Fraction of data for validation.")
    p.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--output_dir", type=Path, default=Path("outputs"), help="Directory for checkpoints and metrics.")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device (cuda or cpu).")
    p.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers.")
    p.add_argument("--no_pin_memory", action="store_true", help="Disable DataLoader pin_memory.")
    p.add_argument("--no_compile", action="store_true", help="Disable torch.compile.")
    p.add_argument("--no_amp", action="store_true", help="Disable Automatic Mixed Precision (AMP).")
    p.add_argument("--resume", type=Path, default=None, help="Path to the checkpoint file to resume from (e.g., outputs/checkpoints/mobilenetv2_last.pt).") # Takes path
    p.add_argument("--profile", action="store_true", help="Run profiler for a few steps and exit.")
    p.add_argument("--freeze_features", action=argparse.BooleanOptionalAction, default=True, help="Freeze feature layers of MobileNetV2.")
    args = p.parse_args()

    # --- Setup ---
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available(): print("Warning: CUDA requested but not available. Switching to CPU."); args.device = "cpu"
    device = torch.device(args.device)
    pin_memory = not args.no_pin_memory and (device.type == 'cuda')
    use_compile = COMPILE_SUPPORTED and not args.no_compile
    use_amp = not args.no_amp and (device.type == 'cuda')

    print("--- MobileNetV2 Training Configuration ---")
    print(f"Device: {device}, CSV: {args.csv}, Batch: {args.batch_size}, Workers: {args.num_workers}")
    print(f"Pin Mem: {pin_memory}, Freeze Feat: {args.freeze_features}, Compile: {use_compile}, AMP: {use_amp}")
    print(f"Resume Ckpt: {args.resume}, Profile: {args.profile}, Output: {args.output_dir}")
    if args.data_root: print(f"Data Root Override: {args.data_root}")
    print("----------------------------------------")

    # --- Load Dataset ---
    print(f"Loading dataset from {args.csv}...")
    try:
        ds = MelSpectrogramPairDataset(args.csv)
        if args.data_root: ds.pairs = remap_pairs(ds.pairs, args.data_root)
        if len(ds) == 0: print("Error: Dataset is empty after loading/remapping."); return
    except FileNotFoundError: print(f"Error: Input CSV file not found at {args.csv}"); return
    except Exception as e: print(f"Error loading dataset: {e}"); return

    # --- Train/Val Split ---
    n_val = int(len(ds) * args.val_split); n_train = len(ds) - n_val
    if n_train <= 0 or n_val <= 0: print(f"Error: Invalid train/validation split."); return
    print(f"Split: {n_train} train, {n_val} validation samples.")
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=(args.num_workers > 0))
    print("DataLoaders created.")

    # --- Initialize Models, Optimizer, Criterion, Scaler ---
    print("Initializing MobileNetV2 model and Classifier...")
    model = MobileNetV2Embedding(freeze=args.freeze_features).to(device)
    clf = PairClassifier().to(device) # Classifier now outputs logits
    # Define params based on freeze flag BEFORE optimizer init
    if args.freeze_features: params_to_optimize = list(clf.parameters()) + list(model.proj.parameters()); print("Optimizing Classifier and MobileNetV2 Projection Layer.")
    else: params_to_optimize = list(model.parameters()) + list(clf.parameters()); print("Optimizing ALL MobileNetV2 and Classifier parameters (fine-tuning).")
    opt = torch.optim.Adam(params_to_optimize, lr=args.lr)
    criterion = nn.BCEWithLogitsLoss() # Use the numerically stable version
    try: scaler = torch.amp.GradScaler(enabled=use_amp)
    except AttributeError: print("Warning: Using deprecated torch.cuda.amp.GradScaler."); scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print("Optimizer, Criterion, Scaler initialized.")

    # --- Checkpoint and Metrics Setup ---
    ckpt_dir = args.output_dir / "checkpoints"; metr_dir = args.output_dir / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True); metr_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt_path = ckpt_dir / "mobilenetv2_last.pt"
    best_ckpt_path = ckpt_dir / "mobilenetv2_best.pt"
    stats_file = metr_dir / "mobilenetv2_train_stats.json"
    print(f"Output directories ensured.")

    start_ep = 1; best_val_loss = float("inf"); patience_ctr = 0; history = []

    # --- Resume Logic (Load state dict BEFORE compiling) ---
    resume_ckpt_path = args.resume
    if resume_ckpt_path and resume_ckpt_path.exists():
        print(f"Attempting to resume from {resume_ckpt_path}")
        try:
            ck = torch.load(resume_ckpt_path, map_location=device)
            def load_state_dict_basic(model_to_load, state_dict, strict=True): # Added strict option
                 is_dp = list(state_dict.keys())[0].startswith('module.')
                 if is_dp: new_state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
                 else: new_state_dict = state_dict
                 model_to_load.load_state_dict(new_state_dict, strict=strict)

            # Load model state (strict=False if features were frozen before but not now, or vice versa)
            load_state_dict_basic(model, ck["model"], strict=not args.freeze_features)
            load_state_dict_basic(clf, ck["clf"])
            try: opt.load_state_dict(ck["optimizer"])
            except ValueError as e: print(f"Warning: Could not load optimizer state dict: {e}. Re-initializing.")
            if "scaler" in ck and use_amp: scaler.load_state_dict(ck["scaler"]); print(" - Loaded AMP scaler state.")
            start_ep = ck.get("epoch", 1) + 1
            best_val_loss = ck.get("best_val_loss", float("inf"))
            patience_ctr = ck.get("patience_ctr", 0)
            if stats_file.exists():
                try:
                    with open(stats_file, 'r') as f: history = json.load(f)
                    if not isinstance(history, list): history = []
                    history = [h for h in history if h.get('epoch', 0) < start_ep]
                    print(f" - Loaded training history up to epoch {start_ep - 1}.")
                except json.JSONDecodeError: print(f" - Warning: Could not parse {stats_file}. Starting fresh history."); history = []
            print(f"Resuming training from epoch {start_ep}. Best validation loss so far: {best_val_loss:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting training from scratch.")
            start_ep = 1; best_val_loss = float("inf"); patience_ctr = 0; history = []
    elif args.resume:
         print(f"Warning: Resume checkpoint specified but not found at {resume_ckpt_path}. Starting from scratch.")

    # --- Apply torch.compile (AFTER potential resume) ---
    if use_compile:
        print("Applying torch.compile...")
        compile_start_time = time.time()
        try: model = torch.compile(model); print(" - MobileNetV2 compiled.")
        except Exception as e: print(f" - Warning: Compiling MobileNetV2 failed: {e}")
        try: clf = torch.compile(clf); print(" - Classifier compiled.")
        except Exception as e: print(f" - Warning: Compiling Classifier failed: {e}")
        print(f"Compilation took {time.time() - compile_start_time:.2f}s")

    # --- Profiler Setup ---
    prof = None
    if args.profile:
        print("Profiler enabled. Will run for ~10 steps and exit.")
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, schedule=torch.profiler.schedule(wait=1, warmup=1, active=8, repeat=1))
        prof.start()

    # --- Training Loop ---
    print(f"\nStarting MobileNetV2 training from epoch {start_ep} to {args.epochs}...")
    training_start_time = time.time()

    for epoch in range(start_ep, args.epochs + 1):
        epoch_start_time = time.time()
        # --- Train ---
        model.train(); clf.train()
        if args.freeze_features:
             for param in model.features.parameters(): param.requires_grad = False

        train_losses = []
        print(f"\n--- Epoch {epoch}/{args.epochs} ---"); print("Training...")
        batch_times = []
        for i, (x1,x2,y) in enumerate(train_loader):
            batch_start_time = time.time()
            x1 = x1.to(device, non_blocking=pin_memory); x2 = x2.to(device, non_blocking=pin_memory)
            y = y.float().to(device, non_blocking=pin_memory)

            opt.zero_grad(set_to_none=True)

            autocast_args = {'device_type': device.type, 'enabled': use_amp}
            try:
                with torch.amp.autocast(**autocast_args):
                    e1 = model(x1); e2 = model(x2)
                    logits = clf(e1,e2); loss = criterion(logits,y)
            except AttributeError:
                 if i == 0: print("Warning: Using deprecated torch.cuda.amp.autocast.")
                 with torch.cuda.amp.autocast(enabled=use_amp):
                    e1 = model(x1); e2 = model(x2)
                    logits = clf(e1,e2); loss = criterion(logits,y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            train_losses.append(loss.item()*y.size(0))
            batch_times.append(time.time() - batch_start_time)

            if prof: prof.step();
            if prof and i >= 9: print("Profiling finished."); break

            if (i + 1) % 50 == 0 or i == len(train_loader) - 1:
                 avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
                 print(f"  Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Avg Batch Time: {avg_batch_time:.3f}s")
                 batch_times = []

        if prof: break # Exit epoch loop after profiling

        if not hasattr(train_loader, 'dataset') or len(train_loader.dataset) == 0: train_loss = float('nan')
        else: num_processed = len(train_losses) * args.batch_size; train_loss = sum(train_losses) / num_processed if num_processed > 0 else 0
        print(f"Epoch {epoch} Training complete. Average Train Loss: {train_loss:.4f}")

        # --- Validate ---
        print("Validating...")
        val_start_time = time.time()
        val_loss, val_auc = evaluate(model, clf, val_loader, criterion, device, use_amp)
        val_end_time = time.time()
        print(f"Epoch {epoch} Validation complete. Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f} (Time: {val_end_time - val_start_time:.2f}s)")

        # --- Log History ---
        history.append({"epoch":epoch,"train_loss":train_loss, "val_loss":val_loss,"val_auc":val_auc})
        try:
            with open(stats_file,"w") as f: json.dump(history,f,indent=2)
        except Exception as e: print(f"Warning: Could not save history to {stats_file}: {e}")

        # --- Checkpointing and Early Stopping ---
        print("Saving last checkpoint...")
        checkpoint = {"epoch": epoch, "model": model.state_dict(), "clf": clf.state_dict(),
                      "optimizer": opt.state_dict(), "best_val_loss": best_val_loss, "patience_ctr": patience_ctr}
        if use_amp: checkpoint["scaler"] = scaler.state_dict()
        try: torch.save(checkpoint, last_ckpt_path)
        except Exception as e: print(f"Warning: Failed to save last checkpoint {last_ckpt_path}: {e}")

        print("Checking for improvement...")
        if not np.isnan(val_loss) and val_loss < best_val_loss:
            print(f"  Validation loss improved ({best_val_loss:.4f} --> {val_loss:.4f}). Saving best model.")
            best_val_loss = val_loss; patience_ctr = 0
            best_checkpoint = {"model": model.state_dict(), "clf": clf.state_dict(),
                               "best_val_loss": best_val_loss, "epoch": epoch}
            try: torch.save(best_checkpoint, best_ckpt_path); print(f"  Saved best checkpoint to {best_ckpt_path}")
            except Exception as e: print(f"Warning: Failed to save best checkpoint {best_ckpt_path}: {e}")
        else:
            patience_ctr += 1
            print(f"  Validation loss did not improve. Patience: {patience_ctr}/{args.patience}")
            if patience_ctr >= args.patience: print(f"Early stopping triggered at epoch {epoch}."); break

        epoch_end_time = time.time()
        print(f"Epoch {epoch} finished in {epoch_end_time - epoch_start_time:.2f} seconds.")

    # --- End of Training ---
    training_end_time = time.time()
    if prof:
        prof.stop()
        print("\n--- Profiler Results ---")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
        print("\n" + prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
    else:
        print("\nMobileNetV2 training finished.")
        print(f"Total training time: {training_end_time - training_start_time:.2f} seconds.")
        print(f"Final history saved to {stats_file}")
        print(f"Last checkpoint: {last_ckpt_path}")
        print(f"Best checkpoint: {best_ckpt_path} (Val Loss: {best_val_loss:.4f})")

if __name__=="__main__":
    main()

# # scripts/train_mobilenetv2.py (Optimized)

# import argparse
# import json
# import os
# import time
# from pathlib import Path

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, random_split
# from sklearn.metrics import roc_auc_score
# from torch.profiler import profile, record_function, ProfilerActivity # For Profiler

# # Import local modules
# try:
#     from datasets.spectrogram_dataset import MelSpectrogramPairDataset
#     from models.mobilenet_embedding import MobileNetV2Embedding
# except ImportError as e:
#     print(f"Error importing local modules: {e}")
#     print("Please ensure datasets/spectrogram_dataset.py and models/mobilenet_embedding.py exist.")
#     exit(1)

# # Check PyTorch version for torch.compile compatibility
# TORCH_MAJOR = int(torch.__version__.split('.')[0])
# TORCH_MINOR = int(torch.__version__.split('.')[1])
# COMPILE_SUPPORTED = (TORCH_MAJOR >= 2)
# if not COMPILE_SUPPORTED:
#     print(f"Warning: torch.compile requires PyTorch 2.0 or later. You have {torch.__version__}. Compilation will be skipped.")

# # --- Pair Classifier (Remains the same) ---
# class PairClassifier(nn.Module):
#     def __init__(self, emb_dim=256, hidden_dim=512):
#         super().__init__()
#         self.fc1 = nn.Linear(emb_dim*2, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
#         self.bn2 = nn.BatchNorm1d(hidden_dim//2)
#         self.out = nn.Linear(hidden_dim//2,1)
#     def forward(self,e1,e2):
#         x = torch.cat([e1,e2],dim=1)
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = F.relu(self.bn2(self.fc2(x)))
#         return torch.sigmoid(self.out(x)).squeeze(1)

# # --- Evaluation Function (Remains mostly the same, added AMP toggle) ---
# def evaluate(model, clf, loader, criterion, device, use_amp=True):
#     model.eval(); clf.eval()
#     total_loss = 0.0
#     all_scores, all_labels = [], []
#     amp_enabled = use_amp and (device.type == 'cuda')

#     with torch.no_grad():
#         for x1,x2,y in loader:
#             x1,x2,y = x1.to(device), x2.to(device), y.float().to(device) # Labels to float

#             with torch.cuda.amp.autocast(enabled=amp_enabled):
#                 e1 = model(x1); e2 = model(x2)
#                 pred = clf(e1,e2)
#                 loss = criterion(pred,y)

#             total_loss += loss.item()*y.size(0)
#             all_scores.append(pred.cpu())
#             all_labels.append(y.cpu())

#     if not hasattr(loader, 'dataset') or len(loader.dataset) == 0:
#          print("Warning: Could not determine dataset size for validation loss calculation.")
#          avg_loss = float('nan')
#     else:
#          avg_loss = total_loss / len(loader.dataset)

#     y_true = torch.cat(all_labels).numpy()
#     y_score= torch.cat(all_scores).numpy()
#     try:
#         auc = roc_auc_score(y_true, y_score)
#     except ValueError as e:
#         print(f"Warning: Could not compute AUC in validation: {e}")
#         auc = 0.0
#     return avg_loss, auc

# # --- Path Remapping Function (Unchanged) ---
# def remap_pairs(pairs, data_root):
#     # This function might not be needed if the _local.csv file already contains
#     # the correct absolute local paths after copying.
#     # Keeping it for flexibility in case CSV paths need adjustment.
#     all_paths = [p for p,_,_ in pairs] + [q for _,q,_ in pairs]
#     if not all_paths:
#         print("Warning: No paths found in pairs list for remapping.")
#         return []
#     try:
#         # This might fail if paths are already absolute local paths and don't share a common root
#         # or if paths point to different locations (e.g. some local, some drive)
#         orig_root = os.path.commonpath(all_paths)
#         print(f"Remapping paths: Original common root = {orig_root}, New root = {data_root}")
#     except ValueError:
#          print(f"Warning: Could not determine common path. Assuming paths in CSV are relative to {data_root} or already absolute.")
#          orig_root = None # Treat paths as potentially relative to data_root

#     new = []
#     skipped_count = 0
#     for p1,p2,l in pairs:
#         try:
#             if orig_root:
#                  rel1 = os.path.relpath(p1, orig_root)
#                  rel2 = os.path.relpath(p2, orig_root)
#             else:
#                  # If no common root, assume p1/p2 are relative paths to be joined with data_root
#                  # Or they might already be the correct full paths
#                  # This logic might need adjustment based on how the CSV paths are formatted
#                  # Let's assume if no common root, they are already correct or relative to data_root
#                  path1_obj = Path(p1)
#                  path2_obj = Path(p2)
#                  # If path is already absolute and starts with data_root, use it directly
#                  if path1_obj.is_absolute() and str(path1_obj).startswith(str(data_root)):
#                       rel1 = path1_obj.relative_to(data_root) # Get relative part for joining
#                  else: # Assume relative or needs joining
#                       rel1 = p1 # Use original path relative to data_root

#                  if path2_obj.is_absolute() and str(path2_obj).startswith(str(data_root)):
#                       rel2 = path2_obj.relative_to(data_root)
#                  else:
#                       rel2 = p2

#             new_p1 = os.path.join(data_root, rel1)
#             new_p2 = os.path.join(data_root, rel2)
#             # Optional: Check if new paths exist
#             # if not Path(new_p1).exists() or not Path(new_p2).exists():
#             #      print(f"Warning: Remapped path does not exist: {new_p1} or {new_p2}. Skipping pair.")
#             #      skipped_count += 1
#             #      continue
#             new.append((new_p1, new_p2, l))
#         except ValueError as e:
#             print(f"Skipping pair due to path issue during remapping: ({p1}, {p2}). Error: {e}")
#             skipped_count += 1
#             continue
#     if skipped_count > 0:
#          print(f"Skipped {skipped_count} pairs during path remapping.")
#     return new

# # --- Main Training Function ---
# def main():
#     p = argparse.ArgumentParser(description="Train MobileNetV2 Speaker Verification Model")
#     # --- Arguments ---
#     p.add_argument("csv", help="Path to the CSV file containing pairs of spectrogram image files (e.g., pairs_spec_train_local.csv)")
#     p.add_argument("--data_root", type=str, default=None,
#                   help="Optional: Local root directory for spectrogram images. Use if paths in CSV need remapping or are relative.")
#     p.add_argument("--epochs", type=int, default=50, help="Max number of training epochs.")
#     p.add_argument("--batch_size", type=int, default=64, help="Training batch size (try increasing this).") # Default increased
#     p.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
#     p.add_argument("--val_split", type=float, default=0.1, help="Fraction of data for validation.")
#     p.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
#     p.add_argument("--seed", type=int, default=42, help="Random seed.")
#     p.add_argument("--output_dir", type=Path, default=Path("outputs"), help="Directory for checkpoints and metrics.")
#     p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device (cuda or cpu).")
#     p.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers (tune this).")
#     p.add_argument("--no_pin_memory", action="store_true", help="Disable DataLoader pin_memory.")
#     p.add_argument("--no_compile", action="store_true", help="Disable torch.compile.")
#     p.add_argument("--no_amp", action="store_true", help="Disable Automatic Mixed Precision (AMP).")
#     p.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint.")
#     p.add_argument("--profile", action="store_true", help="Run profiler for a few steps and exit.") # Profiler flag
#     p.add_argument("--freeze_features", action=argparse.BooleanOptionalAction, default=True, help="Freeze feature layers of MobileNetV2.") # Control freezing

#     args = p.parse_args()

#     # --- Setup ---
#     torch.manual_seed(args.seed)
#     if args.device == "cuda" and not torch.cuda.is_available():
#         print("Warning: CUDA requested but not available. Switching to CPU.")
#         args.device = "cpu"
#     device = torch.device(args.device)
#     pin_memory = not args.no_pin_memory and (device.type == 'cuda')
#     use_compile = COMPILE_SUPPORTED and not args.no_compile
#     use_amp = not args.no_amp and (device.type == 'cuda')

#     print("--- MobileNetV2 Training Configuration ---")
#     print(f"Device: {device}")
#     print(f"Input CSV: {args.csv}")
#     if args.data_root: print(f"Data Root Override: {args.data_root}")
#     print(f"Batch Size: {args.batch_size}")
#     print(f"Num Workers: {args.num_workers}")
#     print(f"Pin Memory: {pin_memory}")
#     print(f"Freeze Features: {args.freeze_features}")
#     print(f"Use torch.compile: {use_compile}")
#     print(f"Use AMP: {use_amp}")
#     print(f"Resume Training: {args.resume}")
#     print(f"Profile Mode: {args.profile}")
#     print(f"Output Directory: {args.output_dir}")
#     print("----------------------------------------")

#     # --- Load Dataset ---
#     print(f"Loading dataset from {args.csv}...")
#     try:
#         ds = MelSpectrogramPairDataset(args.csv)
#         if args.data_root:
#             print(f"Attempting to remap spectrogram paths to root: {args.data_root}")
#             ds.pairs = remap_pairs(ds.pairs, args.data_root)
#             if not ds.pairs:
#                  print("Warning: No valid pairs found after remapping paths.")
#                  # Decide whether to exit or continue with an empty dataset
#                  # exit(1) # Or let it fail later

#         if len(ds) == 0:
#             print("Error: Dataset is empty after loading/remapping. Check CSV and data_root. Exiting.")
#             return
#     except FileNotFoundError:
#         print(f"Error: Input CSV file not found at {args.csv}")
#         return
#     except Exception as e:
#         print(f"Error loading dataset: {e}")
#         return


#     # --- Train/Val Split ---
#     n_val = int(len(ds) * args.val_split)
#     n_train = len(ds) - n_val
#     if n_train <= 0 or n_val <= 0:
#          print(f"Error: Invalid train/validation split from dataset size {len(ds)}.")
#          return
#     print(f"Split: {n_train} train, {n_val} validation samples.")
#     train_ds, val_ds = random_split(
#         ds, [n_train, n_val],
#         generator=torch.Generator().manual_seed(args.seed)
#     )
#     # Add persistent_workers=True
#     train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
#                               num_workers=args.num_workers, pin_memory=pin_memory,
#                               persistent_workers=(args.num_workers > 0))
#     val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
#                             num_workers=args.num_workers, pin_memory=pin_memory,
#                             persistent_workers=(args.num_workers > 0))
#     print("DataLoaders created.")


#     # --- Models, Optimizer, Criterion ---
#     print("Initializing MobileNetV2 model and Classifier...")
#     # Pass freeze argument to model
#     model = MobileNetV2Embedding(freeze=args.freeze_features).to(device)
#     clf = PairClassifier().to(device)

#     # --- Apply torch.compile (if enabled) ---
#     if use_compile:
#         print("Applying torch.compile...")
#         compile_start_time = time.time()
#         try:
#             model = torch.compile(model)
#             print(" - MobileNetV2 compiled.")
#         except Exception as e:
#             print(f" - Warning: Compiling MobileNetV2 failed: {e}")
#         try:
#             clf = torch.compile(clf)
#             print(" - Classifier compiled.")
#         except Exception as e:
#             print(f" - Warning: Compiling Classifier failed: {e}")
#         print(f"Compilation took {time.time() - compile_start_time:.2f}s")

#     # --- Optimizer and Loss ---
#     # Adjust parameters to optimize based on freeze flag
#     if args.freeze_features:
#          params_to_optimize = list(clf.parameters()) + list(model.proj.parameters()) # Only classifier + projection layer
#          print("Optimizing Classifier and MobileNetV2 Projection Layer parameters.")
#     else:
#          params_to_optimize = list(model.parameters()) + list(clf.parameters()) # All parameters
#          print("Optimizing ALL MobileNetV2 and Classifier parameters (fine-tuning).")

#     opt = torch.optim.Adam(params_to_optimize, lr=args.lr)
#     crit = nn.BCELoss()
#     print("Optimizer and Criterion initialized.")

#     # --- AMP GradScaler ---
#     scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
#     print(f"AMP GradScaler initialized (Enabled: {use_amp}).")

#     # --- Checkpoint and Metrics Setup ---
#     ckpt_dir = args.output_dir / "checkpoints"
#     metr_dir = args.output_dir / "metrics"
#     ckpt_dir.mkdir(parents=True, exist_ok=True)
#     metr_dir.mkdir(parents=True, exist_ok=True)
#     print(f"Output directories ensured: Checkpoints='{ckpt_dir}', Metrics='{metr_dir}'")

#     start_ep = 1
#     best_val_loss = float("inf")
#     patience_ctr = 0
#     history = []

#     last_ckpt = ckpt_dir / "mobilenetv2_last.pt" # Consistent naming
#     best_ckpt = ckpt_dir / "mobilenetv2_best.pt"
#     stats_file = metr_dir / "mobilenetv2_train_stats.json"

#     # --- Resume Logic ---
#     if args.resume and last_ckpt.exists():
#         print(f"Attempting to resume from {last_ckpt}")
#         try:
#             ck = torch.load(last_ckpt, map_location=device)
#             def load_state_dict_flexible(model_to_load, state_dict):
#                  if list(state_dict.keys())[0].startswith('module.'):
#                      from collections import OrderedDict
#                      new_state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
#                      model_to_load.load_state_dict(new_state_dict, strict=False) # Use strict=False if only loading partial (e.g. proj layer)
#                  else:
#                      model_to_load.load_state_dict(state_dict, strict=False) # Use strict=False

#             load_state_dict_flexible(model, ck["model"])
#             load_state_dict_flexible(clf, ck["clf"])
#             # Only load optimizer state if parameters match (can be tricky if freeze changed)
#             # Safer to re-initialize optimizer if freeze status might change between runs
#             # Or save/load optimizer state based on freeze flag
#             try:
#                  opt.load_state_dict(ck["optimizer"])
#             except ValueError as e:
#                  print(f"Warning: Could not load optimizer state dict, possibly due to parameter mismatch (e.g., freeze changed): {e}. Re-initializing optimizer state.")
#                  # Optimizer state will be reset to initial

#             if "scaler" in ck and use_amp:
#                  scaler.load_state_dict(ck["scaler"])
#                  print(" - Loaded AMP scaler state.")

#             start_ep = ck.get("epoch", 1) + 1
#             best_val_loss = ck.get("best_val_loss", float("inf"))
#             patience_ctr = ck.get("patience_ctr", 0)

#             if stats_file.exists():
#                 try:
#                     with open(stats_file, 'r') as f: history = json.load(f)
#                     if not isinstance(history, list): history = []
#                     history = [h for h in history if h.get('epoch', 0) < start_ep]
#                     print(f" - Loaded training history up to epoch {start_ep - 1}.")
#                 except json.JSONDecodeError:
#                     print(f" - Warning: Could not parse {stats_file}. Starting fresh history.")
#                     history = []
#             print(f"Resuming training from epoch {start_ep}. Best validation loss so far: {best_val_loss:.4f}")

#         except Exception as e:
#             print(f"Error loading checkpoint: {e}. Starting training from scratch.")
#             start_ep = 1; best_val_loss = float("inf"); patience_ctr = 0; history = []


#     # --- Profiler Setup ---
#     prof = None
#     if args.profile:
#         print("Profiler enabled. Will run for ~10 steps and exit.")
#         prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#                       record_shapes=True, profile_memory=True, schedule=torch.profiler.schedule(wait=1, warmup=1, active=8, repeat=1))
#         prof.start()

#     # --- Training Loop ---
#     print(f"\nStarting MobileNetV2 training from epoch {start_ep} to {args.epochs}...")
#     training_start_time = time.time()

#     for epoch in range(start_ep, args.epochs + 1):
#         epoch_start_time = time.time()
#         # --- Train ---
#         model.train(); clf.train()
#         # Ensure feature layers are frozen/unfrozen correctly each epoch if needed
#         # (though typically set once at init)
#         if args.freeze_features:
#              for param in model.features.parameters():
#                  param.requires_grad = False
#         else:
#              for param in model.features.parameters():
#                   param.requires_grad = True


#         train_losses = []
#         print(f"\n--- Epoch {epoch}/{args.epochs} ---")
#         print("Training...")
#         batch_times = []
#         for i, (x1,x2,y) in enumerate(train_loader):
#             batch_start_time = time.time()
#             x1 = x1.to(device, non_blocking=pin_memory)
#             x2 = x2.to(device, non_blocking=pin_memory)
#             y = y.float().to(device, non_blocking=pin_memory)

#             opt.zero_grad(set_to_none=True)

#             with torch.cuda.amp.autocast(enabled=use_amp):
#                 e1 = model(x1); e2 = model(x2)
#                 pred = clf(e1,e2)
#                 loss = crit(pred,y)

#             scaler.scale(loss).backward()
#             scaler.step(opt)
#             scaler.update()

#             train_losses.append(loss.item()*y.size(0))
#             batch_times.append(time.time() - batch_start_time)

#             # --- Profiler Step ---
#             if prof:
#                 prof.step()
#                 if i >= 9: # Profiling window
#                     print("Profiling finished.")
#                     break

#             # Optional batch progress print
#             if (i + 1) % 50 == 0 or i == len(train_loader) - 1:
#                  avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
#                  print(f"  Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Avg Batch Time: {avg_batch_time:.3f}s")
#                  batch_times = []

#         # --- Exit after profiling if enabled ---
#         if prof:
#             break

#         if not hasattr(train_loader, 'dataset') or len(train_loader.dataset) == 0:
#              print("Warning: Could not determine training dataset size.")
#              train_loss = float('nan')
#         else:
#              num_processed = len(train_losses) * args.batch_size # Approximate
#              train_loss = sum(train_losses) / num_processed if num_processed > 0 else 0
#              # train_loss = sum(train_losses) / len(train_ds) # If full epoch runs

#         print(f"Epoch {epoch} Training complete. Average Train Loss: {train_loss:.4f}")

#         # --- Validate ---
#         print("Validating...")
#         val_start_time = time.time()
#         val_loss, val_auc = evaluate(model, clf, val_loader, crit, device, use_amp)
#         val_end_time = time.time()
#         print(f"Epoch {epoch} Validation complete. Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f} (Time: {val_end_time - val_start_time:.2f}s)")

#         # --- Log History ---
#         history.append({"epoch":epoch,"train_loss":train_loss, "val_loss":val_loss,"val_auc":val_auc})
#         try:
#             with open(stats_file,"w") as f: json.dump(history,f,indent=2)
#         except Exception as e:
#             print(f"Warning: Could not save history to {stats_file}: {e}")

#         # --- Checkpointing and Early Stopping ---
#         print("Saving last checkpoint...")
#         checkpoint = {
#             "epoch": epoch, "model": model.state_dict(), "clf": clf.state_dict(),
#             "optimizer": opt.state_dict(), "best_val_loss": best_val_loss,
#             "patience_ctr": patience_ctr,
#         }
#         if use_amp: checkpoint["scaler"] = scaler.state_dict()
#         try:
#             torch.save(checkpoint, last_ckpt)
#         except Exception as e:
#             print(f"Warning: Failed to save last checkpoint {last_ckpt}: {e}")

#         print("Checking for improvement...")
#         if val_loss < best_val_loss:
#             print(f"  Validation loss improved ({best_val_loss:.4f} --> {val_loss:.4f}). Saving best model.")
#             best_val_loss = val_loss; patience_ctr = 0
#             best_checkpoint = {"model": model.state_dict(), "clf": clf.state_dict(),
#                               "best_val_loss": best_val_loss, "epoch": epoch}
#             try:
#                 torch.save(best_checkpoint, best_ckpt)
#                 print(f"  Saved best checkpoint to {best_ckpt}")
#             except Exception as e:
#                  print(f"Warning: Failed to save best checkpoint {best_ckpt}: {e}")
#         else:
#             patience_ctr += 1
#             print(f"  Validation loss did not improve. Patience: {patience_ctr}/{args.patience}")
#             if patience_ctr >= args.patience:
#                 print(f"Early stopping triggered at epoch {epoch}.")
#                 break

#         epoch_end_time = time.time()
#         print(f"Epoch {epoch} finished in {epoch_end_time - epoch_start_time:.2f} seconds.")

#     # --- End of Training ---
#     training_end_time = time.time()
#     if prof:
#         prof.stop()
#         print("\n--- Profiler Results ---")
#         print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
#         print("\n" + prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
#         # trace_file = f"mobilenet_trace_{time.strftime('%Y%m%d_%H%M%S')}.json"
#         # try:
#         #     prof.export_chrome_trace(trace_file)
#         #     print(f"Profiler trace saved to {trace_file}")
#         # except Exception as e:
#         #     print(f"Failed to save profiler trace: {e}")
#     else:
#         print("\nMobileNetV2 training finished.")
#         print(f"Total training time: {training_end_time - training_start_time:.2f} seconds.")
#         print(f"Final history saved to {stats_file}")
#         print(f"Last checkpoint: {last_ckpt}")
#         print(f"Best checkpoint: {best_ckpt} (Val Loss: {best_val_loss:.4f})")

# if __name__=="__main__":
#     main()

# # # scripts/train_mobilenetv2.py

# # import argparse, json, os
# # from pathlib import Path

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch.utils.data import DataLoader, random_split
# # from sklearn.metrics import roc_auc_score

# # from datasets.spectrogram_dataset import MelSpectrogramPairDataset # Make sure this path is correct
# # from models.mobilenet_embedding import MobileNetV2Embedding # Make sure this path is correct

# # class PairClassifier(nn.Module):
# #     def __init__(self, emb_dim=256, hidden_dim=512):
# #         super().__init__()
# #         self.fc1 = nn.Linear(emb_dim*2, hidden_dim)
# #         self.bn1 = nn.BatchNorm1d(hidden_dim)
# #         self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
# #         self.bn2 = nn.BatchNorm1d(hidden_dim//2)
# #         self.out = nn.Linear(hidden_dim//2,1)
# #     def forward(self,e1,e2):
# #         x = torch.cat([e1,e2],dim=1)
# #         x = F.relu(self.bn1(self.fc1(x)))
# #         x = F.relu(self.bn2(self.fc2(x)))
# #         return torch.sigmoid(self.out(x)).squeeze(1)

# # def evaluate(model, clf, loader, criterion, device):
# #     model.eval(); clf.eval()
# #     losses, scores, labels = [], [], []
# #     with torch.no_grad():
# #         for x1,x2,y in loader:
# #             x1,x2,y = x1.to(device), x2.to(device), y.float().to(device)
# #             e1 = model(x1); e2 = model(x2)
# #             pred = clf(e1,e2)
# #             losses.append(criterion(pred,y).item()*y.size(0))
# #             scores.append(pred.cpu())
# #             labels.append(y.cpu())
# #     avg_loss = sum(losses)/len(loader.dataset)
# #     y_true = torch.cat(labels).numpy()
# #     y_score= torch.cat(scores).numpy()
# #     auc = roc_auc_score(y_true, y_score)
# #     return avg_loss, auc

# # def remap_pairs(pairs, data_root):
# #     # infer original root from common path
# #     all_paths = [p for p,_,_ in pairs] + [q for _,q,_ in pairs]
# #     if not all_paths:
# #         return []
# #     orig_root = os.path.commonpath(all_paths)
# #     print(f"Remapping paths: Original common root = {orig_root}, New root = {data_root}")
# #     new = []
# #     for p1,p2,l in pairs:
# #         try:
# #             rel1 = os.path.relpath(p1, orig_root)
# #             rel2 = os.path.relpath(p2, orig_root)
# #             new.append((os.path.join(data_root, rel1),
# #                         os.path.join(data_root, rel2),
# #                         l))
# #         except ValueError as e:
# #             print(f"Skipping pair due to path issue: ({p1}, {p2}). Error: {e}")
# #             continue
# #     return new

# # def main():
# #     p = argparse.ArgumentParser()
# #     p.add_argument("csv", help="pairs_spec_train.csv")
# #     p.add_argument("--data_root",     type=str,   default=None,
# #                   help="Local root for your spectrogram images (overrides CSV paths)")
# #     p.add_argument("--epochs",        type=int,   default=50)
# #     p.add_argument("--batch_size",    type=int,   default=32)
# #     p.add_argument("--lr",            type=float, default=1e-4)
# #     p.add_argument("--val_split",     type=float, default=0.1)
# #     p.add_argument("--patience",      type=int,   default=5)
# #     p.add_argument("--resume",        action="store_true")
# #     p.add_argument("--seed",          type=int,   default=42)
# #     p.add_argument("--output_dir",    type=Path,  default=Path("outputs"))
# #     p.add_argument("--device",        type=str,   default="cuda")
# #     args = p.parse_args()

# #     torch.manual_seed(args.seed)
# #     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
# #     print(f"Using device: {device}")

# #     # load dataset, optionally remap to local
# #     print(f"Loading dataset from {args.csv}...")
# #     ds = MelSpectrogramPairDataset(args.csv)
# #     if args.data_root:
# #         print(f"Remapping spectrogram paths to root: {args.data_root}")
# #         ds.pairs = remap_pairs(ds.pairs, args.data_root)
# #         if not ds.pairs:
# #              print("Error: No valid pairs found after remapping paths. Exiting.")
# #              return

# #     # train/val split
# #     n_val = int(len(ds) * args.val_split)
# #     n_train = len(ds) - n_val
# #     print(f"Split: {n_train} train, {n_val} validation samples.")
# #     train_ds, val_ds = random_split(
# #         ds, [n_train, n_val],
# #         generator=torch.Generator().manual_seed(args.seed)
# #     )
# #     train_loader = DataLoader(train_ds, batch_size=args.batch_size,
# #                               shuffle=True, num_workers=4, pin_memory=True)
# #     val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
# #                               shuffle=False, num_workers=4, pin_memory=True)
# #     print("DataLoaders created.")

# #     # models, optimizer, criterion
# #     print("Initializing MobileNetV2 model and Classifier...")
# #     model = MobileNetV2Embedding().to(device)
# #     clf   = PairClassifier().to(device)
# #     opt   = torch.optim.Adam(list(model.parameters())+list(clf.parameters()), lr=args.lr)
# #     crit  = nn.BCELoss()
# #     print("Model, Classifier, Optimizer, and Criterion initialized.")

# #     ckpt_dir = args.output_dir/"checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
# #     metr_dir = args.output_dir/"metrics";    metr_dir.mkdir(parents=True, exist_ok=True)
# #     print(f"Output directories prepared: {ckpt_dir}, {metr_dir}")

# #     start_ep, best_val, patience = 1, float("inf"), 0
# #     history = []

# #     last_ckpt = ckpt_dir/"mobilenetv2_last.pt"
# #     best_ckpt = ckpt_dir/"mobilenetv2_best.pt"
# #     stats_file= metr_dir/"mobilenetv2_train_stats.json"

# #     if args.resume and last_ckpt.exists():
# #         print(f"Attempting to resume training from {last_ckpt}")
# #         ck = torch.load(last_ckpt, map_location=device)
# #         model.load_state_dict(ck["model"])
# #         clf.load_state_dict(ck["clf"])
# #         opt.load_state_dict(ck["optimizer"])
# #         start_ep   = ck["epoch"] + 1
# #         best_val    = ck["best_val_loss"]
# #         patience    = ck["patience_ctr"]
# #         if stats_file.exists():
# #             try:
# #                  with open(stats_file, 'r') as f:
# #                     history = json.load(f)
# #                  # Ensure history is a list
# #                  if not isinstance(history, list):
# #                     print(f"Warning: Corrupted history file {stats_file}. Starting fresh history.")
# #                     history = []
# #                  else:
# #                     # Optional: Trim history if it goes beyond the resumed epoch
# #                     history = [h for h in history if h.get('epoch', 0) < start_ep]
# #             except json.JSONDecodeError:
# #                 print(f"Warning: Could not parse {stats_file}. Starting fresh history.")
# #                 history = []
# #         print(f"Resuming MobileNetV2 training from epoch {start_ep}")

# #     print(f"\nStarting MobileNetV2 training from epoch {start_ep} to {args.epochs}...")
# #     for epoch in range(start_ep, args.epochs+1):
# #         model.train(); clf.train()
# #         train_losses = []
# #         print(f"\n--- Epoch {epoch}/{args.epochs} ---")
# #         print("Training MobileNetV2...")
# #         for i, (x1,x2,y) in enumerate(train_loader):
# #             x1,x2,y = x1.to(device), x2.to(device), y.float().to(device)
# #             e1 = model(x1); e2 = model(x2)
# #             pred = clf(e1,e2)
# #             loss = crit(pred,y)
# #             opt.zero_grad(); loss.backward(); opt.step()
# #             train_losses.append(loss.item()*y.size(0))
# #             # Optional batch progress print
# #             # if (i + 1) % 50 == 0:
# #             #      print(f"  Batch {i+1}/{len(train_loader)}, Current Batch Loss: {loss.item():.4f}")

# #         train_loss = sum(train_losses)/len(train_ds)
# #         print(f"Epoch {epoch} Training complete. Average Train Loss: {train_loss:.4f}")

# #         print("Validating MobileNetV2...")
# #         val_loss, val_auc = evaluate(model, clf, val_loader, crit, device)
# #         print(f"Epoch {epoch} Validation complete. Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

# #         history.append({"epoch":epoch,"train_loss":train_loss,
# #                         "val_loss":val_loss,"val_auc":val_auc})
# #         try:
# #             with open(stats_file,"w") as f: json.dump(history,f,indent=2)
# #         except Exception as e:
# #             print(f"Error saving stats file: {e}")


# #         print("Saving checkpoint...")
# #         # always save last
# #         try:
# #             torch.save({
# #                 "epoch":         epoch,
# #                 "model":         model.state_dict(),
# #                 "clf":           clf.state_dict(),
# #                 "optimizer":     opt.state_dict(),
# #                 "best_val_loss": best_val,
# #                 "patience_ctr":  patience
# #             }, last_ckpt)
# #             print(f"  Saved last checkpoint to {last_ckpt}")
# #         except Exception as e:
# #              print(f"Error saving last checkpoint: {e}")


# #         # Original print statement (optional)
# #         # print(f"[MobileNetV2] Ep{epoch}: train={train_loss:.4f} val={val_loss:.4f} auc={val_auc:.4f}")

# #         # best?
# #         print("Checking for improvement...")
# #         if val_loss < best_val:
# #             print(f"  Validation loss improved ({best_val:.4f} --> {val_loss:.4f}). Saving best model.")
# #             best_val = val_loss; patience = 0
# #             try:
# #                 torch.save({
# #                     "model":         model.state_dict(),
# #                     "clf":           clf.state_dict(),
# #                     # Removed optimizer state dict from best checkpoint to save space, optional
# #                     # "optimizer":     opt.state_dict(),
# #                     "best_val_loss": best_val,
# #                     # "patience_ctr":  patience # Not needed in best checkpoint
# #                 }, best_ckpt)
# #                 print(f"  Saved best checkpoint to {best_ckpt}")
# #             except Exception as e:
# #                 print(f"Error saving best checkpoint: {e}")
# #         else:
# #             patience += 1
# #             print(f"  Validation loss did not improve. Patience: {patience}/{args.patience}")
# #             if patience >= args.patience:
# #                 print(f"Early stopping triggered at epoch {epoch}")
# #                 break

# #     print("\nMobileNetV2 Training finished.")
# #     print(f"Final training history saved to {stats_file}")
# #     print(f"Last checkpoint saved to {last_ckpt}")
# #     print(f"Best checkpoint saved to {best_ckpt}")

# # if __name__=="__main__":
# #     main()