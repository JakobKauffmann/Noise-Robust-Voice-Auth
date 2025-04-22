# scripts/train_sincnet.py (Optimized for Preprocessed Data + AMP Loss Fix)

import argparse
import json
import time
from pathlib import Path
import numpy as np # For isnan check

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score
from torch.profiler import profile, record_function, ProfilerActivity # For Profiler

# Import the NEW dataset class for preprocessed data
try:
    # Assuming you saved the class from Part 2 as preprocessed_raw_dataset.py
    from datasets.preprocessed_raw_dataset import RawAudioDatasetPreprocessed
    from models.sincnet import SincNetEmbedding
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Please ensure datasets/preprocessed_raw_dataset.py and models/sincnet.py exist.")
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
        self.fc1 = nn.Linear(emb_dim * 2, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2, 1) # Outputs raw logits

    def forward(self, e1, e2):
        x = torch.cat([e1, e2], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        # REMOVED torch.sigmoid() here - output raw logits
        return self.out(x).squeeze(1)


# --- Evaluation Function (MODIFIED: Handles logits) ---
def evaluate(model, clf, loader, criterion, device, use_amp=True):
    model.eval()
    clf.eval()
    total_loss = 0.0
    all_scores = []
    all_labels = []
    amp_enabled = use_amp and (device.type == 'cuda')

    with torch.no_grad():
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.float().to(device) # Labels to float

            # Use updated torch.amp namespace if possible
            autocast_args = {'device_type': device.type, 'enabled': amp_enabled}
            try:
                with torch.amp.autocast(**autocast_args):
                    e1 = model(x1)
                    e2 = model(x2)
                    logits = clf(e1, e2) # Get logits from the modified classifier
                    loss = criterion(logits, y) # Criterion handles logits now
            except AttributeError:
                 with torch.cuda.amp.autocast(enabled=amp_enabled): # Fallback
                    e1 = model(x1)
                    e2 = model(x2)
                    logits = clf(e1, e2)
                    loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            all_scores.append(logits.cpu())
            all_labels.append(y.cpu()) # Keep original labels (0 or 1)

    if not hasattr(loader, 'dataset') or len(loader.dataset) == 0:
         print("Warning: Could not determine dataset size for validation loss calculation.")
         avg_loss = float('nan')
    else:
         avg_loss = total_loss / len(loader.dataset)

    y_true = torch.cat(all_labels).numpy()
    y_score = torch.cat(all_scores).numpy() # Pass logits directly to roc_auc_score
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError as e:
        print(f"Warning: Could not compute AUC in validation: {e}")
        auc = 0.0
    return avg_loss, auc


# --- Main Training Function ---
def main():
    p = argparse.ArgumentParser(description="Train SincNet Speaker Verification Model using Preprocessed Data")
    # --- Arguments ---
    p.add_argument("preprocessed_csv", help="Path to the CSV file containing pairs of preprocessed .npy audio files (e.g., pairs_raw_train_preprocessed_local.csv)")
    p.add_argument("--epochs", type=int, default=50, help="Max number of training epochs.")
    p.add_argument("--batch_size", type=int, default=64, help="Training batch size (try increasing this).")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    p.add_argument("--val_split", type=float, default=0.1, help="Fraction of data for validation.")
    p.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--output_dir", type=Path, default=Path("outputs"), help="Directory for checkpoints and metrics.")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device (cuda or cpu).")
    p.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers (tune this).")
    p.add_argument("--no_pin_memory", action="store_true", help="Disable DataLoader pin_memory.")
    p.add_argument("--no_compile", action="store_true", help="Disable torch.compile.")
    p.add_argument("--no_amp", action="store_true", help="Disable Automatic Mixed Precision (AMP).")
    p.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint.")
    p.add_argument("--profile", action="store_true", help="Run profiler for a few steps and exit.")
    args = p.parse_args()

    # --- Setup ---
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Switching to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    pin_memory = not args.no_pin_memory and (device.type == 'cuda')
    use_compile = COMPILE_SUPPORTED and not args.no_compile
    use_amp = not args.no_amp and (device.type == 'cuda')

    print("--- SincNet Training Configuration ---")
    print(f"Device: {device}")
    print(f"Preprocessed CSV: {args.preprocessed_csv}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Num Workers: {args.num_workers}")
    print(f"Pin Memory: {pin_memory}")
    print(f"Use torch.compile: {use_compile}")
    print(f"Use AMP: {use_amp}")
    print(f"Resume Training: {args.resume}")
    print(f"Profile Mode: {args.profile}")
    print(f"Output Directory: {args.output_dir}")
    print("------------------------------------")

    # --- Dataset and Split ---
    print("Loading dataset using RawAudioDatasetPreprocessed...")
    try:
        full_ds = RawAudioDatasetPreprocessed(args.preprocessed_csv)
        if len(full_ds) == 0:
             print("Error: Dataset is empty after loading. Check CSV path and content.")
             return
    except FileNotFoundError:
         print(f"Error: Preprocessed CSV file not found at {args.preprocessed_csv}")
         return
    except Exception as e:
         print(f"Error loading dataset: {e}")
         return

    n_val = int(len(full_ds) * args.val_split)
    n_train = len(full_ds) - n_val
    if n_train <= 0 or n_val <= 0:
         print(f"Error: Invalid train/validation split from dataset size {len(full_ds)}.")
         return
    print(f"Split: {n_train} train, {n_val} validation samples.")
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_memory,
                              persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin_memory,
                            persistent_workers=(args.num_workers > 0))
    print("DataLoaders created.")

    # --- Models, Optimizer, Criterion ---
    print("Initializing SincNet model and Classifier...")
    model = SincNetEmbedding().to(device)
    clf = PairClassifier().to(device) # Classifier now outputs logits

    # --- Apply torch.compile (if enabled) ---
    if use_compile:
        print("Applying torch.compile...")
        compile_start_time = time.time()
        try: model = torch.compile(model); print(" - SincNet compiled.")
        except Exception as e: print(f" - Warning: Compiling SincNet failed: {e}")
        try: clf = torch.compile(clf); print(" - Classifier compiled.")
        except Exception as e: print(f" - Warning: Compiling Classifier failed: {e}")
        print(f"Compilation took {time.time() - compile_start_time:.2f}s")

    # --- Optimizer and Loss (MODIFIED: Use BCEWithLogitsLoss) ---
    optimizer = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss() # Use the numerically stable version
    print("Optimizer and BCEWithLogitsLoss Criterion initialized.")

    # --- AMP GradScaler (MODIFIED: Use torch.amp namespace) ---
    try:
         scaler = torch.amp.GradScaler(device_type=device.type, enabled=use_amp)
    except AttributeError:
         print("Warning: Using deprecated torch.cuda.amp.GradScaler.")
         scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"AMP GradScaler initialized (Enabled: {use_amp}).")

    # --- Checkpoint and Metrics Setup ---
    ckpt_dir = args.output_dir / "checkpoints"
    metr_dir = args.output_dir / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metr_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directories ensured: Checkpoints='{ckpt_dir}', Metrics='{metr_dir}'")

    start_ep = 1
    best_val_loss = float("inf")
    patience_ctr = 0
    history = []

    last_ckpt = ckpt_dir / "sincnet_last.pt"
    best_ckpt = ckpt_dir / "sincnet_best.pt"
    stats_file = metr_dir / "sincnet_train_stats.json"

    # --- Resume Logic ---
    if args.resume and last_ckpt.exists():
        print(f"Attempting to resume from {last_ckpt}")
        try:
            ck = torch.load(last_ckpt, map_location=device)
            def load_state_dict_flexible(model_to_load, state_dict):
                 is_dp_or_ddp = list(state_dict.keys())[0].startswith('module.')
                 is_compiled = list(state_dict.keys())[0].startswith('_orig_mod.')
                 if is_compiled:
                      from collections import OrderedDict
                      new_state_dict = OrderedDict()
                      for k, v in state_dict.items():
                          name = k.replace('_orig_mod.', '')
                          if name.startswith('module.'): name = name[7:]
                          new_state_dict[name] = v
                      model_to_load.load_state_dict(new_state_dict)
                 elif is_dp_or_ddp:
                      from collections import OrderedDict
                      new_state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
                      model_to_load.load_state_dict(new_state_dict)
                 else:
                      model_to_load.load_state_dict(state_dict)

            load_state_dict_flexible(model, ck["model"])
            load_state_dict_flexible(clf, ck["clf"])
            optimizer.load_state_dict(ck["optimizer"])

            if "scaler" in ck and use_amp:
                 scaler.load_state_dict(ck["scaler"])
                 print(" - Loaded AMP scaler state.")

            start_ep = ck.get("epoch", 1) + 1
            best_val_loss = ck.get("best_val_loss", float("inf"))
            patience_ctr = ck.get("patience_ctr", 0)

            if stats_file.exists():
                try:
                    with open(stats_file, 'r') as f: history = json.load(f)
                    if not isinstance(history, list): history = []
                    history = [h for h in history if h.get('epoch', 0) < start_ep]
                    print(f" - Loaded training history up to epoch {start_ep - 1}.")
                except json.JSONDecodeError:
                    print(f" - Warning: Could not parse {stats_file}. Starting fresh history.")
                    history = []
            print(f"Resuming training from epoch {start_ep}. Best validation loss so far: {best_val_loss:.4f}")

        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting training from scratch.")
            start_ep = 1; best_val_loss = float("inf"); patience_ctr = 0; history = []

    # --- Profiler Setup ---
    prof = None
    if args.profile:
        print("Profiler enabled. Will run for ~10 steps and exit.")
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                       record_shapes=True, profile_memory=True, schedule=torch.profiler.schedule(wait=1, warmup=1, active=8, repeat=1))
        prof.start()

    # --- Training Loop ---
    print(f"\nStarting SincNet training from epoch {start_ep} to {args.epochs}...")
    training_start_time = time.time()

    for epoch in range(start_ep, args.epochs + 1):
        epoch_start_time = time.time()
        # --- Train ---
        model.train(); clf.train()
        train_losses = []
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        print("Training...")
        batch_times = []
        for i, (x1, x2, y) in enumerate(train_loader):
            batch_start_time = time.time()
            x1 = x1.to(device, non_blocking=pin_memory)
            x2 = x2.to(device, non_blocking=pin_memory)
            y = y.float().to(device, non_blocking=pin_memory) # Labels to float

            optimizer.zero_grad(set_to_none=True)

            # Autocast context (MODIFIED: Use torch.amp namespace)
            autocast_args = {'device_type': device.type, 'enabled': use_amp}
            try:
                 with torch.amp.autocast(**autocast_args):
                     e1 = model(x1)
                     e2 = model(x2)
                     logits = clf(e1, e2) # Get logits
                     loss = criterion(logits, y) # Use logits with BCEWithLogitsLoss
            except AttributeError:
                 if i == 0: print("Warning: Using deprecated torch.cuda.amp.autocast.")
                 with torch.cuda.amp.autocast(enabled=use_amp):
                     e1 = model(x1)
                     e2 = model(x2)
                     logits = clf(e1, e2)
                     loss = criterion(logits, y)

            # Scaled backward pass
            scaler.scale(loss).backward()
            # Optimizer step
            scaler.step(optimizer)
            # Update scaler
            scaler.update()

            train_losses.append(loss.item() * y.size(0))
            batch_times.append(time.time() - batch_start_time)

             # --- Profiler Step ---
            if prof:
                prof.step()
                if i >= 9: # Profiling window
                    print("Profiling finished.")
                    break

            # Optional batch progress print
            if (i + 1) % 50 == 0 or i == len(train_loader) - 1:
                 avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
                 print(f"  Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Avg Batch Time: {avg_batch_time:.3f}s")
                 batch_times = []

        # --- Exit after profiling if enabled ---
        if prof: break

        if not hasattr(train_loader, 'dataset') or len(train_loader.dataset) == 0:
             print("Warning: Could not determine training dataset size.")
             train_loss = float('nan')
        else:
             num_processed = len(train_losses) * args.batch_size
             train_loss = sum(train_losses) / num_processed if num_processed > 0 else 0

        print(f"Epoch {epoch} Training complete. Average Train Loss: {train_loss:.4f}")

        # --- Validate ---
        print("Validating...")
        val_start_time = time.time()
        val_loss, val_auc = evaluate(model, clf, val_loader, criterion, device, use_amp)
        val_end_time = time.time()
        print(f"Epoch {epoch} Validation complete. Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f} (Time: {val_end_time - val_start_time:.2f}s)")

        # --- Log History ---
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc})
        try:
            with open(stats_file, "w") as f: json.dump(history, f, indent=2)
        except Exception as e: print(f"Warning: Could not save history to {stats_file}: {e}")

        # --- Checkpointing and Early Stopping ---
        print("Saving last checkpoint...")
        checkpoint = {
            "epoch": epoch, "model": model.state_dict(), "clf": clf.state_dict(),
            "optimizer": optimizer.state_dict(), "best_val_loss": best_val_loss,
            "patience_ctr": patience_ctr,
        }
        if use_amp: checkpoint["scaler"] = scaler.state_dict()
        try: torch.save(checkpoint, last_ckpt)
        except Exception as e: print(f"Warning: Failed to save last checkpoint {last_ckpt}: {e}")

        print("Checking for improvement...")
        if not np.isnan(val_loss) and val_loss < best_val_loss: # Check for NaN val_loss
            print(f"  Validation loss improved ({best_val_loss:.4f} --> {val_loss:.4f}). Saving best model.")
            best_val_loss = val_loss; patience_ctr = 0
            best_checkpoint = {"model": model.state_dict(), "clf": clf.state_dict(),
                               "best_val_loss": best_val_loss, "epoch": epoch}
            try:
                torch.save(best_checkpoint, best_ckpt)
                print(f"  Saved best checkpoint to {best_ckpt}")
            except Exception as e: print(f"Warning: Failed to save best checkpoint {best_ckpt}: {e}")
        else:
            patience_ctr += 1
            print(f"  Validation loss did not improve. Patience: {patience_ctr}/{args.patience}")
            if patience_ctr >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        epoch_end_time = time.time()
        print(f"Epoch {epoch} finished in {epoch_end_time - epoch_start_time:.2f} seconds.")

    # --- End of Training ---
    training_end_time = time.time()
    if prof:
        prof.stop()
        print("\n--- Profiler Results ---")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
        print("\n" + prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
        # trace_file = f"sincnet_trace_{time.strftime('%Y%m%d_%H%M%S')}.json"
        # try: prof.export_chrome_trace(trace_file); print(f"Profiler trace saved to {trace_file}")
        # except Exception as e: print(f"Failed to save profiler trace: {e}")
    else:
        print("\nSincNet training finished.")
        print(f"Total training time: {training_end_time - training_start_time:.2f} seconds.")
        print(f"Final history saved to {stats_file}")
        print(f"Last checkpoint: {last_ckpt}")
        print(f"Best checkpoint: {best_ckpt} (Val Loss: {best_val_loss:.4f})")

if __name__ == "__main__":
    main()

# # scripts/train_sincnet.py (Optimized for Preprocessed Data)

# import argparse
# import json
# import time
# from pathlib import Path

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, random_split
# from sklearn.metrics import roc_auc_score
# from torch.profiler import profile, record_function, ProfilerActivity # For Profiler

# # Import the NEW dataset class for preprocessed data
# try:
#     from datasets.preprocessed_raw_dataset import RawAudioDatasetPreprocessed
#     from models.sincnet import SincNetEmbedding
# except ImportError as e:
#     print(f"Error importing local modules: {e}")
#     print("Please ensure datasets/preprocessed_raw_dataset.py and models/sincnet.py exist.")
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
#         # Adjusted hidden dim based on potential embedding size
#         self.fc1 = nn.Linear(emb_dim * 2, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
#         self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
#         self.out = nn.Linear(hidden_dim // 2, 1)

#     def forward(self, e1, e2):
#         x = torch.cat([e1, e2], dim=1)
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = F.relu(self.bn2(self.fc2(x)))
#         return torch.sigmoid(self.out(x)).squeeze(1)


# # --- Evaluation Function (Remains mostly the same, added AMP toggle) ---
# def evaluate(model, clf, loader, criterion, device, use_amp=True):
#     model.eval()
#     clf.eval()
#     total_loss = 0.0
#     all_scores = []
#     all_labels = []
#     amp_enabled = use_amp and (device.type == 'cuda')

#     with torch.no_grad():
#         for x1, x2, y in loader:
#             x1, x2, y = x1.to(device), x2.to(device), y.float().to(device) # Labels to float

#             with torch.cuda.amp.autocast(enabled=amp_enabled):
#                 e1 = model(x1)
#                 e2 = model(x2)
#                 preds = clf(e1, e2)
#                 loss = criterion(preds, y)

#             total_loss += loss.item() * y.size(0)
#             all_scores.append(preds.cpu())
#             all_labels.append(y.cpu()) # Keep labels as they are (likely int)

#     # Ensure loader.dataset is accessible and correct
#     if not hasattr(loader, 'dataset') or len(loader.dataset) == 0:
#          print("Warning: Could not determine dataset size for validation loss calculation.")
#          avg_loss = float('nan') # Or handle appropriately
#     else:
#          avg_loss = total_loss / len(loader.dataset)

#     y_true = torch.cat(all_labels).numpy()
#     y_score = torch.cat(all_scores).numpy()
#     try:
#         auc = roc_auc_score(y_true, y_score)
#     except ValueError as e:
#         print(f"Warning: Could not compute AUC in validation: {e}")
#         auc = 0.0
#     return avg_loss, auc


# # --- Main Training Function ---
# def main():
#     p = argparse.ArgumentParser(description="Train SincNet Speaker Verification Model using Preprocessed Data")
#     # --- Arguments ---
#     p.add_argument("preprocessed_csv", help="Path to the CSV file containing pairs of preprocessed .npy audio files (e.g., pairs_raw_train_preprocessed.csv)")
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

#     print("--- SincNet Training Configuration ---")
#     print(f"Device: {device}")
#     print(f"Preprocessed CSV: {args.preprocessed_csv}")
#     print(f"Batch Size: {args.batch_size}")
#     print(f"Num Workers: {args.num_workers}")
#     print(f"Pin Memory: {pin_memory}")
#     print(f"Use torch.compile: {use_compile}")
#     print(f"Use AMP: {use_amp}")
#     print(f"Resume Training: {args.resume}")
#     print(f"Profile Mode: {args.profile}")
#     print(f"Output Directory: {args.output_dir}")
#     print("------------------------------------")

#     # --- Dataset and Split ---
#     print("Loading dataset using RawAudioDatasetPreprocessed...")
#     try:
#         # Use the NEW dataset class
#         full_ds = RawAudioDatasetPreprocessed(args.preprocessed_csv)
#         if len(full_ds) == 0:
#              print("Error: Dataset is empty after loading. Check CSV path and content.")
#              return
#     except FileNotFoundError:
#          print(f"Error: Preprocessed CSV file not found at {args.preprocessed_csv}")
#          return
#     except Exception as e:
#          print(f"Error loading dataset: {e}")
#          return

#     n_val = int(len(full_ds) * args.val_split)
#     n_train = len(full_ds) - n_val
#     if n_train <= 0 or n_val <= 0:
#          print(f"Error: Invalid train/validation split from dataset size {len(full_ds)}.")
#          return
#     print(f"Split: {n_train} train, {n_val} validation samples.")
#     train_ds, val_ds = random_split(
#         full_ds, [n_train, n_val],
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
#     print("Initializing SincNet model and Classifier...")
#     model = SincNetEmbedding().to(device) # Assuming default params are okay
#     clf = PairClassifier().to(device) # Assuming default params are okay

#     # --- Apply torch.compile (if enabled) ---
#     # Compile *before* loading state_dict if resuming, potentially safer
#     # Or compile here and load state_dict into compiled model (usually works)
#     if use_compile:
#         print("Applying torch.compile...")
#         compile_start_time = time.time()
#         try:
#             model = torch.compile(model)
#             print(" - SincNet compiled.")
#         except Exception as e:
#             print(f" - Warning: Compiling SincNet failed: {e}")
#         try:
#             clf = torch.compile(clf)
#             print(" - Classifier compiled.")
#         except Exception as e:
#             print(f" - Warning: Compiling Classifier failed: {e}")
#         print(f"Compilation took {time.time() - compile_start_time:.2f}s")

#     # --- Optimizer and Loss ---
#     optimizer = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()), lr=args.lr)
#     criterion = nn.BCELoss()
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

#     last_ckpt = ckpt_dir / "sincnet_last.pt" # Consistent naming
#     best_ckpt = ckpt_dir / "sincnet_best.pt"
#     stats_file = metr_dir / "sincnet_train_stats.json"

#     # --- Resume Logic ---
#     if args.resume and last_ckpt.exists():
#         print(f"Attempting to resume from {last_ckpt}")
#         try:
#             # Load checkpoint onto the correct device
#             ck = torch.load(last_ckpt, map_location=device)

#             # Load state dicts
#             # Handle potential DataParallel wrapping if saved that way
#             def load_state_dict_flexible(model_to_load, state_dict):
#                  if list(state_dict.keys())[0].startswith('module.'):
#                      from collections import OrderedDict
#                      new_state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
#                      model_to_load.load_state_dict(new_state_dict)
#                  else:
#                      model_to_load.load_state_dict(state_dict)

#             load_state_dict_flexible(model, ck["model"])
#             load_state_dict_flexible(clf, ck["clf"])
#             optimizer.load_state_dict(ck["optimizer"])

#             if "scaler" in ck and use_amp:
#                  scaler.load_state_dict(ck["scaler"])
#                  print(" - Loaded AMP scaler state.")

#             start_ep = ck.get("epoch", 1) + 1
#             best_val_loss = ck.get("best_val_loss", float("inf"))
#             patience_ctr = ck.get("patience_ctr", 0)

#             # Load history
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
#                        record_shapes=True, profile_memory=True, schedule=torch.profiler.schedule(wait=1, warmup=1, active=8, repeat=1))
#         prof.start()

#     # --- Training Loop ---
#     print(f"\nStarting SincNet training from epoch {start_ep} to {args.epochs}...")
#     training_start_time = time.time()

#     for epoch in range(start_ep, args.epochs + 1):
#         epoch_start_time = time.time()
#         # --- Train ---
#         model.train(); clf.train()
#         train_losses = []
#         print(f"\n--- Epoch {epoch}/{args.epochs} ---")
#         print("Training...")
#         batch_times = []
#         for i, (x1, x2, y) in enumerate(train_loader):
#             batch_start_time = time.time()
#             x1 = x1.to(device, non_blocking=pin_memory)
#             x2 = x2.to(device, non_blocking=pin_memory)
#             y = y.float().to(device, non_blocking=pin_memory) # Labels to float for BCELoss

#             optimizer.zero_grad(set_to_none=True)

#             # Autocast context
#             with torch.cuda.amp.autocast(enabled=use_amp):
#                 e1 = model(x1)
#                 e2 = model(x2)
#                 preds = clf(e1, e2)
#                 loss = criterion(preds, y)

#             # Scaled backward pass
#             scaler.scale(loss).backward()
#             # Optimizer step
#             scaler.step(optimizer)
#             # Update scaler
#             scaler.update()

#             train_losses.append(loss.item() * y.size(0))
#             batch_times.append(time.time() - batch_start_time)

#              # --- Profiler Step ---
#             if prof:
#                 prof.step()
#                 # Stop after profiling window completes
#                 if i >= 9: # (wait=1 + warmup=1 + active=8 = 10 steps total)
#                     print("Profiling finished.")
#                     break # Exit inner loop

#             # Optional batch progress print
#             if (i + 1) % 50 == 0 or i == len(train_loader) - 1:
#                  avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
#                  print(f"  Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Avg Batch Time: {avg_batch_time:.3f}s")
#                  batch_times = []

#         # --- Exit after profiling if enabled ---
#         if prof:
#             break # Exit outer loop

#         # Ensure train_ds accessible and correct length
#         if not hasattr(train_loader, 'dataset') or len(train_loader.dataset) == 0:
#              print("Warning: Could not determine training dataset size.")
#              train_loss = float('nan')
#         else:
#             # Calculate train_loss based on actual number of samples processed if loop broke early
#             num_processed = len(train_losses) * args.batch_size # Approximate
#             train_loss = sum(train_losses) / num_processed if num_processed > 0 else 0
#             # More accurate: train_loss = sum(train_losses) / len(train_ds) # Use if full epoch runs

#         print(f"Epoch {epoch} Training complete. Average Train Loss: {train_loss:.4f}")

#         # --- Validate ---
#         print("Validating...")
#         val_start_time = time.time()
#         val_loss, val_auc = evaluate(model, clf, val_loader, criterion, device, use_amp)
#         val_end_time = time.time()
#         print(f"Epoch {epoch} Validation complete. Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f} (Time: {val_end_time - val_start_time:.2f}s)")

#         # --- Log History ---
#         history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc})
#         try:
#             with open(stats_file, "w") as f: json.dump(history, f, indent=2)
#         except Exception as e:
#             print(f"Warning: Could not save history to {stats_file}: {e}")

#         # --- Checkpointing and Early Stopping ---
#         print("Saving last checkpoint...")
#         checkpoint = {
#             "epoch": epoch, "model": model.state_dict(), "clf": clf.state_dict(),
#             "optimizer": optimizer.state_dict(), "best_val_loss": best_val_loss,
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
#                                "best_val_loss": best_val_loss, "epoch": epoch}
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
#         # Optional: Save trace for chrome://tracing
#         # trace_file = f"sincnet_trace_{time.strftime('%Y%m%d_%H%M%S')}.json"
#         # try:
#         #     prof.export_chrome_trace(trace_file)
#         #     print(f"Profiler trace saved to {trace_file}")
#         # except Exception as e:
#         #     print(f"Failed to save profiler trace: {e}")
#     else:
#         print("\nSincNet training finished.")
#         print(f"Total training time: {training_end_time - training_start_time:.2f} seconds.")
#         print(f"Final history saved to {stats_file}")
#         print(f"Last checkpoint: {last_ckpt}")
#         print(f"Best checkpoint: {best_ckpt} (Val Loss: {best_val_loss:.4f})")

# if __name__ == "__main__":
#     main()


# # # scripts/train_sincnet.py

# # import argparse
# # import json
# # from pathlib import Path

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch.utils.data import DataLoader, random_split
# # from sklearn.metrics import roc_auc_score

# # from datasets.raw_dataset import RawAudioPairDataset # Make sure this path is correct
# # from models.sincnet import SincNetEmbedding # Make sure this path is correct


# # class PairClassifier(nn.Module):
# #     def __init__(self, emb_dim=256, hidden_dim=512):
# #         super().__init__()
# #         self.fc1 = nn.Linear(emb_dim * 2, hidden_dim)
# #         self.bn1 = nn.BatchNorm1d(hidden_dim)
# #         self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
# #         self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
# #         self.out = nn.Linear(hidden_dim // 2, 1)

# #     def forward(self, e1, e2):
# #         x = torch.cat([e1, e2], dim=1)
# #         x = F.relu(self.bn1(self.fc1(x)))
# #         x = F.relu(self.bn2(self.fc2(x)))
# #         return torch.sigmoid(self.out(x)).squeeze(1)


# # def evaluate(model, clf, loader, criterion, device):
# #     model.eval()
# #     clf.eval()
# #     losses = []
# #     all_scores = []
# #     all_labels = []
# #     with torch.no_grad():
# #         for x1, x2, y in loader:
# #             x1, x2, y = x1.to(device), x2.to(device), y.float().to(device)
# #             e1 = model(x1)
# #             e2 = model(x2)
# #             preds = clf(e1, e2)
# #             loss = criterion(preds, y)
# #             losses.append(loss.item() * y.size(0))
# #             all_scores.append(preds.cpu())
# #             all_labels.append(y.cpu())
# #     avg_loss = sum(losses) / len(loader.dataset)
# #     y_true = torch.cat(all_labels).numpy()
# #     y_score = torch.cat(all_scores).numpy()
# #     auc = roc_auc_score(y_true, y_score)
# #     return avg_loss, auc


# # def main():
# #     p = argparse.ArgumentParser()
# #     p.add_argument("csv", help="pairs_raw_train.csv")
# #     p.add_argument("--epochs",      type=int,   default=50)
# #     p.add_argument("--batch_size",  type=int,   default=32)
# #     p.add_argument("--lr",          type=float, default=1e-4)
# #     p.add_argument("--val_split",   type=float, default=0.1,
# #                   help="Fraction of data for validation")
# #     p.add_argument("--patience",    type=int,   default=5,
# #                   help="Early stopping patience")
# #     p.add_argument("--seed",        type=int,   default=42)
# #     p.add_argument("--output_dir",  type=Path,  default=Path("outputs"))
# #     p.add_argument("--device",      type=str,   default="cuda")
# #     args = p.parse_args()

# #     torch.manual_seed(args.seed)
# #     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
# #     print(f"Using device: {device}")

# #     # Dataset and split
# #     print("Loading dataset...")
# #     full_ds = RawAudioPairDataset(args.csv)
# #     n_val = int(len(full_ds) * args.val_split)
# #     n_train = len(full_ds) - n_val
# #     train_ds, val_ds = random_split(
# #         full_ds, [n_train, n_val],
# #         generator=torch.Generator().manual_seed(args.seed)
# #     )
# #     print(f"Split: {n_train} train, {n_val} validation samples.")
# #     train_loader = DataLoader(train_ds, batch_size=args.batch_size,
# #                               shuffle=True,  num_workers=4, pin_memory=True)
# #     val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
# #                               shuffle=False, num_workers=4, pin_memory=True)
# #     print("DataLoaders created.")

# #     # Models, optimizer, criterion
# #     print("Initializing SincNet model and Classifier...")
# #     model = SincNetEmbedding().to(device)
# #     clf   = PairClassifier().to(device)
# #     optimizer = torch.optim.Adam(list(model.parameters()) +
# #                                  list(clf.parameters()), lr=args.lr)
# #     criterion = nn.BCELoss()
# #     print("Model, Classifier, Optimizer, and Criterion initialized.")

# #     # Prepare output dirs
# #     ckpt_dir = args.output_dir / "checkpoints"
# #     metr_dir = args.output_dir / "metrics"
# #     ckpt_dir.mkdir(parents=True, exist_ok=True)
# #     metr_dir.mkdir(parents=True, exist_ok=True)
# #     print(f"Output directories prepared: {ckpt_dir}, {metr_dir}")

# #     best_val_loss = float("inf")
# #     patience_ctr = 0
# #     history = []

# #     print(f"\nStarting SincNet training for {args.epochs} epochs...")
# #     for epoch in range(1, args.epochs + 1):
# #         # --- train ---
# #         model.train(); clf.train()
# #         train_losses = []
# #         print(f"\n--- Epoch {epoch}/{args.epochs} ---")
# #         print("Training...")
# #         for i, (x1, x2, y) in enumerate(train_loader):
# #             x1, x2, y = x1.to(device), x2.to(device), y.float().to(device)
# #             e1 = model(x1)
# #             e2 = model(x2)
# #             preds = clf(e1, e2)
# #             loss = criterion(preds, y)
# #             optimizer.zero_grad()
# #             loss.backward()
# #             optimizer.step()
# #             train_losses.append(loss.item() * y.size(0))
# #             # Optional batch progress print
# #             if (i + 1) % 50 == 0:
# #                  print(f"  Batch {i+1}/{len(train_loader)}, Current Batch Loss: {loss.item():.4f}")

# #         train_loss = sum(train_losses) / len(train_ds)
# #         print(f"Epoch {epoch} Training complete. Average Train Loss: {train_loss:.4f}")

# #         # --- validate ---
# #         print("Validating...")
# #         val_loss, val_auc = evaluate(model, clf, val_loader, criterion, device)
# #         print(f"Epoch {epoch} Validation complete. Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

# #         # Log
# #         history.append({
# #             "epoch": epoch,
# #             "train_loss": train_loss,
# #             "val_loss":   val_loss,
# #             "val_auc":    val_auc
# #         })
# #         # Original print statement (optional)
# #         # print(f"[SincNet] Ep {epoch}: train_loss={train_loss:.4f} "
# #         #       f"val_loss={val_loss:.4f} val_auc={val_auc:.4f}")

# #         # --- Early stopping and checkpointing ---
# #         print("Checking for improvement...")
# #         if val_loss < best_val_loss:
# #             print(f"  Validation loss improved ({best_val_loss:.4f} --> {val_loss:.4f}). Saving best model.")
# #             best_val_loss = val_loss
# #             patience_ctr = 0
# #             best_ckpt_path = ckpt_dir / "sincnet_best.pt"
# #             torch.save({
# #                 "model": model.state_dict(),
# #                 "clf":   clf.state_dict()
# #             }, best_ckpt_path)
# #             print(f"  Saved best model checkpoint to {best_ckpt_path}")
# #         else:
# #             patience_ctr += 1
# #             print(f"  Validation loss did not improve. Patience: {patience_ctr}/{args.patience}")
# #             if patience_ctr >= args.patience:
# #                 print(f"Early stopping triggered at epoch {epoch}")
# #                 break

# #     # Save full history
# #     stats_path = metr_dir / "sincnet_train_stats.json"
# #     print(f"\nTraining finished. Saving training history to {stats_path}")
# #     with open(stats_path, "w") as fp:
# #         json.dump(history, fp, indent=2)
# #     print("SincNet training script finished.")


# # if __name__ == "__main__":
# #     main()
# # # # scripts/train_sincnet.py

# # # import argparse
# # # import json
# # # from pathlib import Path

# # # import torch
# # # import torch.nn as nn
# # # import torch.nn.functional as F
# # # from torch.utils.data import DataLoader, random_split
# # # from sklearn.metrics import roc_auc_score

# # # from datasets.raw_dataset import RawAudioPairDataset
# # # from models.sincnet import SincNetEmbedding


# # # class PairClassifier(nn.Module):
# # #     def __init__(self, emb_dim=256, hidden_dim=512):
# # #         super().__init__()
# # #         self.fc1 = nn.Linear(emb_dim * 2, hidden_dim)
# # #         self.bn1 = nn.BatchNorm1d(hidden_dim)
# # #         self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
# # #         self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
# # #         self.out = nn.Linear(hidden_dim // 2, 1)

# # #     def forward(self, e1, e2):
# # #         x = torch.cat([e1, e2], dim=1)
# # #         x = F.relu(self.bn1(self.fc1(x)))
# # #         x = F.relu(self.bn2(self.fc2(x)))
# # #         return torch.sigmoid(self.out(x)).squeeze(1)


# # # def evaluate(model, clf, loader, criterion, device):
# # #     model.eval()
# # #     clf.eval()
# # #     losses = []
# # #     all_scores = []
# # #     all_labels = []
# # #     with torch.no_grad():
# # #         for x1, x2, y in loader:
# # #             x1, x2, y = x1.to(device), x2.to(device), y.float().to(device)
# # #             e1 = model(x1)
# # #             e2 = model(x2)
# # #             preds = clf(e1, e2)
# # #             loss = criterion(preds, y)
# # #             losses.append(loss.item() * y.size(0))
# # #             all_scores.append(preds.cpu())
# # #             all_labels.append(y.cpu())
# # #     avg_loss = sum(losses) / len(loader.dataset)
# # #     y_true = torch.cat(all_labels).numpy()
# # #     y_score = torch.cat(all_scores).numpy()
# # #     auc = roc_auc_score(y_true, y_score)
# # #     return avg_loss, auc


# # # def main():
# # #     p = argparse.ArgumentParser()
# # #     p.add_argument("csv", help="pairs_raw_train.csv")
# # #     p.add_argument("--epochs",      type=int,   default=50)
# # #     p.add_argument("--batch_size",  type=int,   default=32)
# # #     p.add_argument("--lr",          type=float, default=1e-4)
# # #     p.add_argument("--val_split",   type=float, default=0.1,
# # #                    help="Fraction of data for validation")
# # #     p.add_argument("--patience",    type=int,   default=5,
# # #                    help="Early stopping patience")
# # #     p.add_argument("--seed",        type=int,   default=42)
# # #     p.add_argument("--output_dir",  type=Path,  default=Path("outputs"))
# # #     p.add_argument("--device",      type=str,   default="cuda")
# # #     args = p.parse_args()

# # #     torch.manual_seed(args.seed)
# # #     device = torch.device(args.device if torch.cuda.is_available() else "cpu")

# # #     # Dataset and split
# # #     full_ds = RawAudioPairDataset(args.csv)
# # #     n_val = int(len(full_ds) * args.val_split)
# # #     n_train = len(full_ds) - n_val
# # #     train_ds, val_ds = random_split(
# # #         full_ds, [n_train, n_val],
# # #         generator=torch.Generator().manual_seed(args.seed)
# # #     )
# # #     train_loader = DataLoader(train_ds, batch_size=args.batch_size,
# # #                               shuffle=True,  num_workers=4, pin_memory=True)
# # #     val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
# # #                               shuffle=False, num_workers=4, pin_memory=True)

# # #     # Models, optimizer, criterion
# # #     model = SincNetEmbedding().to(device)
# # #     clf   = PairClassifier().to(device)
# # #     optimizer = torch.optim.Adam(list(model.parameters()) +
# # #                                  list(clf.parameters()), lr=args.lr)
# # #     criterion = nn.BCELoss()

# # #     # Prepare output dirs
# # #     ckpt_dir = args.output_dir / "checkpoints"
# # #     metr_dir = args.output_dir / "metrics"
# # #     ckpt_dir.mkdir(parents=True, exist_ok=True)
# # #     metr_dir.mkdir(parents=True, exist_ok=True)

# # #     best_val_loss = float("inf")
# # #     patience_ctr = 0
# # #     history = []

# # #     for epoch in range(1, args.epochs + 1):
# # #         # --- train ---
# # #         model.train(); clf.train()
# # #         train_losses = []
# # #         for x1, x2, y in train_loader:
# # #             x1, x2, y = x1.to(device), x2.to(device), y.float().to(device)
# # #             e1 = model(x1)
# # #             e2 = model(x2)
# # #             preds = clf(e1, e2)
# # #             loss = criterion(preds, y)
# # #             optimizer.zero_grad()
# # #             loss.backward()
# # #             optimizer.step()
# # #             train_losses.append(loss.item() * y.size(0))
# # #         train_loss = sum(train_losses) / len(train_ds)

# # #         # --- validate ---
# # #         val_loss, val_auc = evaluate(model, clf, val_loader, criterion, device)

# # #         # Log
# # #         history.append({
# # #             "epoch": epoch,
# # #             "train_loss": train_loss,
# # #             "val_loss":   val_loss,
# # #             "val_auc":    val_auc
# # #         })
# # #         print(f"[SincNet] Ep {epoch}: train_loss={train_loss:.4f} "
# # #               f"val_loss={val_loss:.4f} val_auc={val_auc:.4f}")

# # #         # Early stopping
# # #         if val_loss < best_val_loss:
# # #             best_val_loss = val_loss
# # #             patience_ctr = 0
# # #             torch.save({
# # #                 "model": model.state_dict(),
# # #                 "clf":   clf.state_dict()
# # #             }, ckpt_dir / "sincnet_best.pt")
# # #         else:
# # #             patience_ctr += 1
# # #             if patience_ctr >= args.patience:
# # #                 print(f"Early stopping at epoch {epoch}")
# # #                 break

# # #     # Save full history
# # #     with open(metr_dir / "sincnet_train_stats.json", "w") as fp:
# # #         json.dump(history, fp, indent=2)


# # # if __name__ == "__main__":
# # #     main()