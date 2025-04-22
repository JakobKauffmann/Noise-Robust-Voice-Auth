# scripts/train_fusion.py (Optimized + AMP Loss + Compile/Resume Fix)

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
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.metrics import roc_auc_score
from torch.profiler import profile, record_function, ProfilerActivity # For Profiler

# Import local modules
try:
    # Use the preprocessed dataset for raw audio
    from datasets.preprocessed_raw_dataset import RawAudioDatasetPreprocessed
    from datasets.spectrogram_dataset import MelSpectrogramPairDataset
    from models.sincnet import SincNetEmbedding
    from models.mobilenet_embedding import MobileNetV2Embedding
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Please ensure datasets/ and models/ directories are in PYTHONPATH.")
    exit(1)

# Check PyTorch version for torch.compile compatibility
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
COMPILE_SUPPORTED = (TORCH_MAJOR >= 2)
if not COMPILE_SUPPORTED:
    print(f"Warning: torch.compile requires PyTorch 2.0 or later. Compilation will be skipped.")

# --- Fusion Dataset (Uses Preprocessed Raw Dataset) ---
class FusionDataset(Dataset):
    """Combines Preprocessed RawAudio and Spectrogram datasets."""
    def __init__(self, raw_preprocessed_csv, spec_csv, spec_root=None):
        print(f"Initializing FusionDataset:")
        print(f" Raw Preprocessed CSV: '{raw_preprocessed_csv}'")
        print(f" Spectrogram CSV: '{spec_csv}'")
        self.raw = RawAudioDatasetPreprocessed(raw_preprocessed_csv)
        self.spec = MelSpectrogramPairDataset(spec_csv)
        # Remapping logic for spectrograms (if needed)
        if spec_root:
            print(f"Remapping spectrogram paths to root: {spec_root}")
            allp = self.spec.pairs
            if not allp: print("Warning: Spectrogram pairs list empty before remapping.")
            else:
                try:
                    path_list = [p for p,_,_ in allp] + [q for _,q,_ in allp]
                    if not path_list: orig_root = None; print("Warning: No valid paths in spec pairs.")
                    else: orig_root = os.path.commonpath(path_list)
                except ValueError: orig_root = None; print("Warning: Could not determine common path for spectrograms.")
                new_pairs = []; skipped_count = 0
                for p1, p2, l in allp:
                    try:
                        if orig_root: rel1=os.path.relpath(p1, orig_root); rel2=os.path.relpath(p2, orig_root)
                        else:
                             path1_obj = Path(p1); path2_obj = Path(p2)
                             if path1_obj.is_absolute() and str(path1_obj).startswith(str(spec_root)): rel1 = path1_obj.relative_to(spec_root)
                             else: rel1 = p1
                             if path2_obj.is_absolute() and str(path2_obj).startswith(str(spec_root)): rel2 = path2_obj.relative_to(spec_root)
                             else: rel2 = p2
                        new_p1 = os.path.join(spec_root, rel1); new_p2 = os.path.join(spec_root, rel2)
                        new_pairs.append((new_p1, new_p2, l))
                    except ValueError as e: skipped_count += 1
                    except Exception as e: skipped_count += 1; print(f"Error remapping spec pair ({p1}, {p2}): {e}")
                if skipped_count > 0: print(f"Warning: Skipped {skipped_count} spec pairs during remapping.")
                self.spec.pairs = new_pairs
                print(f"Remapping complete. Number of spectrogram pairs: {len(self.spec.pairs)}")

        if len(self.raw) != len(self.spec): print(f"CRITICAL WARNING: Mismatch in dataset lengths! Raw Prep: {len(self.raw)}, Spec: {len(self.spec)}.")
        elif len(self.raw) == 0: print("CRITICAL WARNING: Datasets have length 0.")
        else: print(f"Datasets loaded successfully. Length: {len(self.raw)}")

    def __len__(self): return min(len(self.raw), len(self.spec))
    def __getitem__(self, idx):
        if idx >= len(self): raise IndexError("Index out of bounds")
        w1, w2, y = self.raw[idx] # Gets preprocessed tensors
        i1, i2, _ = self.spec[idx] # Gets spec images (loaded by its dataset)
        return w1, w2, i1, i2, y

# --- Fusion Classifier (MODIFIED: Removed final sigmoid) ---
class FusionClassifier(nn.Module):
    """Classifier for fused SincNet and MobileNet embeddings."""
    def __init__(self, emb_dim=256, hidden_dim=512):
        super().__init__()
        in_dim = 2 * 2 * emb_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2, 1) # Outputs raw logits

    def forward(self, a1, a2, s1, s2):
        e1 = torch.cat([a1, s1], dim=1)
        e2 = torch.cat([a2, s2], dim=1)
        x = torch.cat([e1, e2], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        # REMOVED torch.sigmoid() here - output raw logits
        return self.out(x).squeeze(1)

# --- Evaluation Function (MODIFIED: Handles logits) ---
def evaluate(sinc, spec, clf, loader, crit, device, use_amp=True):
    """Evaluates the fusion model."""
    sinc.eval(); spec.eval(); clf.eval()
    total_loss = 0.0
    all_scores, all_labels = [], []
    amp_enabled = use_amp and (device.type == 'cuda')

    with torch.no_grad():
        for w1, w2, i1, i2, y in loader:
            w1, w2, i1, i2, y = (w1.to(device), w2.to(device), i1.to(device), i2.to(device), y.float().to(device))

            autocast_args = {'device_type': device.type, 'enabled': amp_enabled}
            try:
                 with torch.amp.autocast(**autocast_args):
                     a1, a2 = sinc(w1), sinc(w2)
                     s1, s2 = spec(i1), spec(i2)
                     logits = clf(a1, a2, s1, s2) # Get logits
                     loss = crit(logits, y) # Use logits
            except AttributeError:
                 with torch.cuda.amp.autocast(enabled=amp_enabled): # Fallback
                     a1, a2 = sinc(w1), sinc(w2)
                     s1, s2 = spec(i1), spec(i2)
                     logits = clf(a1, a2, s1, s2)
                     loss = crit(logits, y)

            total_loss += loss.item() * y.size(0)
            all_scores.append(logits.cpu())
            all_labels.append(y.cpu())

    if not hasattr(loader, 'dataset') or len(loader.dataset) == 0: avg_loss = float('nan')
    else: avg_loss = total_loss / len(loader.dataset)
    y_true = torch.cat(all_labels).numpy(); y_score = torch.cat(all_scores).numpy()
    try: auc = roc_auc_score(y_true, y_score)
    except ValueError as e: print(f"Warning: Could not compute AUC in validation: {e}"); auc = 0.0
    return avg_loss, auc

# --- Helper to load state dicts robustly ---
def load_state_dict_flexible(model_to_load, state_dict, strict=True):
    is_dp = list(state_dict.keys())[0].startswith('module.')
    is_comp = list(state_dict.keys())[0].startswith('_orig_mod.')
    if is_comp: new_state_dict = OrderedDict((k.replace('_orig_mod.', ''), v) for k, v in state_dict.items())
    else: new_state_dict = state_dict
    if is_dp: new_state_dict = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in new_state_dict.items())

    try:
        model_to_load.load_state_dict(new_state_dict, strict=strict)
    except RuntimeError as e:
         print(f"Warning: State dict loading error (strict={strict}): {e}. Attempting non-strict load.")
         if strict: # Only retry if strict failed
              model_to_load.load_state_dict(new_state_dict, strict=False)


# --- Main Training Function ---
def main():
    p = argparse.ArgumentParser(description="Train Fusion Speaker Verification Model")
    # --- Arguments ---
    p.add_argument("raw_preprocessed_csv", help="Path to training CSV for PREPROCESSED raw audio pairs (.npy paths)")
    p.add_argument("spec_csv", help="Path to training CSV for spectrogram pairs (.jpg paths)")
    p.add_argument("--spec_data_root", type=str, default=None, help="Optional: Local root directory for spectrogram images.")
    p.add_argument("--sincnet_ckpt", type=Path, default=Path("outputs/checkpoints/sincnet_best.pt"), help="Path to SincNet best checkpoint.")
    p.add_argument("--mobilenet_ckpt", type=Path, default=Path("outputs/checkpoints/mobilenetv2_best.pt"), help="Path to MobileNetV2 best checkpoint.")
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
    p.add_argument("--resume", type=Path, default=None, help="Path to the checkpoint file to resume from (e.g., outputs/checkpoints/fusion_last.pt).") # Takes path
    p.add_argument("--profile", action="store_true", help="Run profiler for a few steps and exit.")
    p.add_argument("--freeze_embedders", action=argparse.BooleanOptionalAction, default=True, help="Freeze SincNet and MobileNetV2 weights.")

    args = p.parse_args()

    # --- Setup ---
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available(): print("Warning: CUDA requested but not available. Switching to CPU."); args.device = "cpu"
    device = torch.device(args.device)
    pin_memory = not args.no_pin_memory and (device.type == 'cuda')
    use_compile = COMPILE_SUPPORTED and not args.no_compile
    use_amp = not args.no_amp and (device.type == 'cuda')

    print("--- Fusion Training Configuration ---")
    print(f"Device: {device}, Raw CSV: {args.raw_preprocessed_csv}, Spec CSV: {args.spec_csv}")
    print(f"Batch: {args.batch_size}, Workers: {args.num_workers}, Pin Mem: {pin_memory}")
    print(f"SincNet Ckpt: {args.sincnet_ckpt}, MobileNet Ckpt: {args.mobilenet_ckpt}")
    print(f"Freeze Embedders: {args.freeze_embedders}, Compile: {use_compile}, AMP: {use_amp}")
    print(f"Resume Ckpt: {args.resume}, Profile: {args.profile}, Output: {args.output_dir}")
    print("-----------------------------------")

    # --- Load Dataset ---
    print("Loading Fusion dataset...")
    try:
        full_ds = FusionDataset(args.raw_preprocessed_csv, args.spec_csv, args.spec_data_root)
        if len(full_ds) == 0: print("Error: Dataset is empty."); return
    except FileNotFoundError as e: print(f"Error: Input CSV not found: {e}"); return
    except Exception as e: print(f"Error loading dataset: {e}"); return

    # --- Train/Val Split ---
    n_val = int(len(full_ds) * args.val_split); n_train = len(full_ds) - n_val
    if n_train <= 0 or n_val <= 0: print(f"Error: Invalid train/validation split."); return
    print(f"Split: {n_train} train, {n_val} validation samples.")
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=(args.num_workers > 0))
    print("DataLoaders created.")

    # --- Initialize Models ---
    print("Initializing models...")
    sinc = SincNetEmbedding().to(device)
    # Initialize MobileNet with correct freeze state for loading checkpoint
    # If fine-tuning, load weights into a non-frozen model
    spec = MobileNetV2Embedding(freeze=False).to(device)
    clf = FusionClassifier().to(device) # Classifier outputs logits

    # --- Load Pre-trained Weights ---
    print(f"Loading SincNet weights from: {args.sincnet_ckpt}")
    if args.sincnet_ckpt.exists():
        try: sinc_ck = torch.load(args.sincnet_ckpt, map_location=device); load_state_dict_flexible(sinc, sinc_ck["model"]); print(" - SincNet weights loaded.")
        except Exception as e: print(f" - Error loading SincNet weights: {e}.")
    else: print(f" - Warning: SincNet checkpoint not found.")

    print(f"Loading MobileNetV2 weights from: {args.mobilenet_ckpt}")
    if args.mobilenet_ckpt.exists():
         try: spec_ck = torch.load(args.mobilenet_ckpt, map_location=device); load_state_dict_flexible(spec, spec_ck["model"], strict=False); print(" - MobileNetV2 weights loaded.") # Use strict=False as best ckpt might only have proj layer if frozen during its training
         except Exception as e: print(f" - Error loading MobileNetV2 weights: {e}.")
    else: print(f" - Warning: MobileNetV2 checkpoint not found.")

    # --- Freeze Embedders AFTER loading weights if requested ---
    if args.freeze_embedders:
        print("Freezing SincNet and MobileNetV2 embedder weights.")
        for param in sinc.parameters(): param.requires_grad = False
        for param in spec.parameters(): param.requires_grad = False
    else:
        print("SincNet and MobileNetV2 embedders will be fine-tuned.")

    # --- Optimizer, Criterion, Scaler ---
    params_to_optimize = list(clf.parameters())
    if not args.freeze_embedders:
        params_to_optimize.extend(list(sinc.parameters()))
        params_to_optimize.extend(list(spec.parameters()))
    print(f"Optimizing {len(params_to_optimize)} parameters.")
    opt = torch.optim.Adam(params_to_optimize, lr=args.lr)
    criterion = nn.BCEWithLogitsLoss() # Use the numerically stable version
    try: scaler = torch.amp.GradScaler(enabled=use_amp)
    except AttributeError: print("Warning: Using deprecated torch.cuda.amp.GradScaler."); scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print("Optimizer, Criterion, Scaler initialized.")

    # --- Checkpoint and Metrics Setup ---
    ckpt_dir = args.output_dir / "checkpoints"; metr_dir = args.output_dir / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True); metr_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt_path = ckpt_dir / "fusion_last.pt"
    best_ckpt_path = ckpt_dir / "fusion_best.pt"
    stats_fn = metr_dir / "fusion_train_stats.json"
    print(f"Output directories ensured.")

    start_ep = 1; best_val_loss = float("inf"); patience_ctr = 0; history = []

    # --- Resume Logic (Load state dict BEFORE compiling) ---
    resume_ckpt_path = args.resume
    if resume_ckpt_path and resume_ckpt_path.exists():
        print(f"Attempting to resume from {resume_ckpt_path}")
        try:
            ck = torch.load(resume_ckpt_path, map_location=device)
            load_state_dict_flexible(clf, ck["clf"]) # Always load classifier
            # Load embedders only if they were being trained in the checkpoint
            if "sinc" in ck and not args.freeze_embedders: load_state_dict_flexible(sinc, ck["sinc"])
            if "spec" in ck and not args.freeze_embedders: load_state_dict_flexible(spec, ck["spec"], strict=False) # Non-strict for MobileNet potentially

            opt.load_state_dict(ck["optimizer"]) # Try loading optimizer
            if "scaler" in ck and use_amp: scaler.load_state_dict(ck["scaler"]); print(" - Loaded AMP scaler state.")
            start_ep = ck.get("epoch", 1) + 1
            best_val_loss = ck.get("best_val_loss", float("inf"))
            patience_ctr = ck.get("patience_ctr", 0)
            if stats_fn.exists():
                try:
                    with open(stats_fn, 'r') as f: history = json.load(f)
                    if not isinstance(history, list): history = []
                    history = [h for h in history if h.get('epoch', 0) < start_ep]
                    print(f" - Loaded training history up to epoch {start_ep - 1}.")
                except json.JSONDecodeError: print(f" - Warning: Could not parse {stats_fn}. Starting fresh history."); history = []
            print(f"Resuming training from epoch {start_ep}. Best validation loss so far: {best_val_loss:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting training from scratch.")
            start_ep = 1; best_val_loss = float("inf"); patience_ctr = 0; history = []
    elif args.resume:
         print(f"Warning: Resume checkpoint specified but not found at {resume_ckpt_path}. Starting from scratch.")


    # --- Apply torch.compile (AFTER potential resume and freezing) ---
    if use_compile:
        print("Applying torch.compile...")
        compile_start_time = time.time()
        try:
             if not args.freeze_embedders: sinc = torch.compile(sinc); print(" - SincNet compiled.")
        except Exception as e: print(f" - Warning: Compiling SincNet failed: {e}")
        try:
             if not args.freeze_embedders: spec = torch.compile(spec); print(" - MobileNetV2 compiled.")
        except Exception as e: print(f" - Warning: Compiling MobileNetV2 failed: {e}")
        try: clf = torch.compile(clf); print(" - FusionClassifier compiled.")
        except Exception as e: print(f" - Warning: Compiling FusionClassifier failed: {e}")
        print(f"Compilation took {time.time() - compile_start_time:.2f}s")


    # --- Profiler Setup ---
    prof = None
    if args.profile:
        print("Profiler enabled. Will run for ~10 steps and exit.")
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, schedule=torch.profiler.schedule(wait=1, warmup=1, active=8, repeat=1))
        prof.start()

    # --- Training Loop ---
    print(f"\nStarting Fusion model training from epoch {start_ep} to {args.epochs}...")
    training_start_time = time.time()

    for epoch in range(start_ep, args.epochs + 1):
        epoch_start_time = time.time()
        # --- Train ---
        clf.train()
        if not args.freeze_embedders: sinc.train(); spec.train()
        else: sinc.eval(); spec.eval() # Keep embedders in eval mode if frozen

        train_losses = []
        print(f"\n--- Epoch {epoch}/{args.epochs} ---"); print("Training...")
        batch_times = []
        for i, (w1, w2, i1, i2, y) in enumerate(train_loader):
            batch_start_time = time.time()
            w1, w2, i1, i2, y = (w1.to(device, non_blocking=pin_memory), w2.to(device, non_blocking=pin_memory),
                                 i1.to(device, non_blocking=pin_memory), i2.to(device, non_blocking=pin_memory),
                                 y.float().to(device, non_blocking=pin_memory))

            opt.zero_grad(set_to_none=True)

            autocast_args = {'device_type': device.type, 'enabled': use_amp}
            try:
                 with torch.amp.autocast(**autocast_args):
                     a1, a2 = sinc(w1), sinc(w2)
                     s1, s2 = spec(i1), spec(i2)
                     logits = clf(a1, a2, s1, s2) # Get logits
                     loss = criterion(logits, y) # Use logits
            except AttributeError:
                 if i == 0: print("Warning: Using deprecated torch.cuda.amp.autocast.")
                 with torch.cuda.amp.autocast(enabled=use_amp):
                     a1, a2 = sinc(w1), sinc(w2)
                     s1, s2 = spec(i1), spec(i2)
                     logits = clf(a1, a2, s1, s2)
                     loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            train_losses.append(loss.item() * y.size(0))
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
        val_loss, val_auc = evaluate(sinc, spec, clf, val_loader, criterion, device, use_amp)
        val_end_time = time.time()
        print(f"Epoch {epoch} Validation complete. Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f} (Time: {val_end_time - val_start_time:.2f}s)")

        # --- Log History ---
        history.append({"epoch":epoch,"train_loss":train_loss, "val_loss":val_loss,"val_auc":val_auc})
        try:
            with open(stats_fn,"w") as f: json.dump(history,f,indent=2)
        except Exception as e: print(f"Warning: Could not save history to {stats_fn}: {e}")

        # --- Checkpointing and Early Stopping ---
        print("Saving last checkpoint...")
        checkpoint = {"epoch": epoch, "clf": clf.state_dict(), "optimizer": opt.state_dict(),
                      "best_val_loss": best_val_loss, "patience_ctr": patience_ctr}
        if not args.freeze_embedders:
             checkpoint["sinc"] = sinc.state_dict()
             checkpoint["spec"] = spec.state_dict()
        if use_amp: checkpoint["scaler"] = scaler.state_dict()
        try: torch.save(checkpoint, last_ckpt_path)
        except Exception as e: print(f"Warning: Failed to save last checkpoint {last_ckpt_path}: {e}")

        print("Checking for improvement...")
        if not np.isnan(val_loss) and val_loss < best_val_loss:
            print(f"  Validation loss improved ({best_val_loss:.4f} --> {val_loss:.4f}). Saving best model.")
            best_val_loss = val_loss; patience_ctr = 0
            best_checkpoint = {"clf": clf.state_dict(), "best_val_loss": best_val_loss, "epoch": epoch}
            if not args.freeze_embedders:
                 best_checkpoint["sinc"] = sinc.state_dict()
                 best_checkpoint["spec"] = spec.state_dict()
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
        print("\nFusion model training finished.")
        print(f"Total training time: {training_end_time - training_start_time:.2f} seconds.")
        print(f"Final history saved to {stats_fn}")
        print(f"Last checkpoint: {last_ckpt_path}")
        print(f"Best checkpoint: {best_ckpt_path} (Val Loss: {best_val_loss:.4f})")

if __name__=="__main__":
    main()

# # scripts/train_fusion.py (Optimized with torch.compile and AMP)

# import argparse
# import json
# import os
# import time # For timing compilation
# from pathlib import Path

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, random_split, Dataset
# from sklearn.metrics import roc_auc_score

# # Import local modules (ensure paths are correct in your environment)
# try:
#     from datasets.raw_dataset import RawAudioPairDataset
#     from datasets.spectrogram_dataset import MelSpectrogramPairDataset
#     from models.sincnet import SincNetEmbedding
#     from models.mobilenet_embedding import MobileNetV2Embedding
# except ImportError as e:
#     print(f"Error importing local modules: {e}")
#     print("Please ensure datasets/ and models/ directories are in PYTHONPATH")
#     exit(1)

# # Check PyTorch version for torch.compile compatibility
# TORCH_MAJOR = int(torch.__version__.split('.')[0])
# TORCH_MINOR = int(torch.__version__.split('.')[1])
# COMPILE_SUPPORTED = (TORCH_MAJOR >= 2)
# if not COMPILE_SUPPORTED:
#     print(f"Warning: torch.compile requires PyTorch 2.0 or later. You have {torch.__version__}. Compilation will be skipped.")

# # --- Fusion Dataset ---
# class FusionDataset(Dataset):
#     """Combines RawAudio and Spectrogram datasets for fusion training."""
#     def __init__(self, raw_csv, spec_csv, spec_root=None):
#         print(f"Initializing FusionDataset: Raw CSV='{raw_csv}', Spec CSV='{spec_csv}'")
#         self.raw = RawAudioPairDataset(raw_csv)
#         self.spec = MelSpectrogramPairDataset(spec_csv)

#         if spec_root:
#             print(f"Remapping spectrogram paths to root: {spec_root}")
#             allp = self.spec.pairs
#             if not allp:
#                 print("Warning: Spectrogram dataset pairs list is empty before remapping.")
#                 self.spec.pairs = [] # Ensure it's an empty list
#             else:
#                 try:
#                     # Attempt to find common path, handle potential errors if paths are diverse
#                     path_list_for_commonpath = [p for p, _, _ in allp] + [q for _, q, _ in allp]
#                     if not path_list_for_commonpath:
#                          orig_root = None # No paths to find common root from
#                          print("Warning: No valid paths found in spectrogram pairs to determine original root.")
#                     else:
#                          orig_root = os.path.commonpath(path_list_for_commonpath)
#                          print(f"Inferred original common path for spectrograms: {orig_root}")

#                 except ValueError:
#                     print("Warning: Could not determine common path for spectrograms (paths might be diverse or relative). Assuming paths are usable as is relative to spec_root.")
#                     orig_root = None # Fallback if common path fails

#                 new_pairs = []
#                 skipped_count = 0
#                 for p1, p2, l in allp:
#                     try:
#                         # Construct new paths based on whether orig_root was found
#                         if orig_root:
#                             rel1 = os.path.relpath(p1, orig_root)
#                             rel2 = os.path.relpath(p2, orig_root)
#                         else:
#                             # If no common root, assume p1/p2 might be relative to something else,
#                             # or maybe they are already absolute paths that just need the root prepended
#                             # This part might need adjustment based on actual CSV content
#                             rel1 = Path(p1).name # Safest fallback: just use filename
#                             rel2 = Path(p2).name
#                             print(f"Warning: Using fallback relative path logic for {p1}, {p2}")

#                         new_p1 = os.path.join(spec_root, rel1)
#                         new_p2 = os.path.join(spec_root, rel2)
#                         new_pairs.append((new_p1, new_p2, l))

#                     except ValueError as e:
#                         print(f"Skipping pair due to path issue: ({p1}, {p2}). Error: {e}")
#                         skipped_count += 1
#                         continue

#                 if skipped_count > 0:
#                     print(f"Warning: Skipped {skipped_count} spectrogram pairs due to path issues during remapping.")
#                 self.spec.pairs = new_pairs
#                 print(f"Remapping complete. Number of spectrogram pairs: {len(self.spec.pairs)}")

#         # Crucial length check
#         if len(self.raw) != len(self.spec):
#             print(f"CRITICAL WARNING: Mismatch in dataset lengths after loading/remapping! Raw: {len(self.raw)}, Spec: {len(self.spec)}. Training will likely fail or produce incorrect results.")
#             # Consider raising an error here: raise ValueError("Dataset length mismatch")
#         elif len(self.raw) == 0:
#              print("CRITICAL WARNING: Both datasets have length 0. No data to train on.")
#         else:
#             print(f"Raw and Spectrogram datasets loaded successfully. Length: {len(self.raw)}")

#     def __len__(self):
#         # Return the minimum length to avoid index errors if mismatch occurs
#         return min(len(self.raw), len(self.spec))

#     def __getitem__(self, idx):
#         # Add explicit check to prevent index error if lengths mismatch
#         if idx >= len(self.raw) or idx >= len(self.spec):
#              raise IndexError(f"Index {idx} out of bounds for dataset lengths (Raw: {len(self.raw)}, Spec: {len(self.spec)})")

#         w1, w2, y = self.raw[idx]
#         i1, i2, _ = self.spec[idx] # Label 'y' should be the same, ignore spec label
#         return w1, w2, i1, i2, y

# # --- Fusion Classifier ---
# class FusionClassifier(nn.Module):
#     """Classifier for fused SincNet and MobileNet embeddings."""
#     def __init__(self, emb_dim=256, hidden_dim=512):
#         super().__init__()
#         in_dim = 2 * 2 * emb_dim # Concatenated pair: (a1, s1, a2, s2)
#         self.fc1 = nn.Linear(in_dim, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
#         self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
#         self.out = nn.Linear(hidden_dim // 2, 1)

#     def forward(self, a1, a2, s1, s2):
#         # a1, a2: SincNet embeddings [B, emb_dim]
#         # s1, s2: MobileNet embeddings [B, emb_dim]
#         e1 = torch.cat([a1, s1], dim=1) # [B, 2*emb_dim]
#         e2 = torch.cat([a2, s2], dim=1) # [B, 2*emb_dim]
#         x = torch.cat([e1, e2], dim=1)  # [B, 4*emb_dim]
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = F.relu(self.bn2(self.fc2(x)))
#         # Output raw logits for BCEWithLogitsLoss, or sigmoid for BCELoss
#         # Using sigmoid here assuming BCELoss later
#         return torch.sigmoid(self.out(x)).squeeze(1)

# # --- Evaluation Function ---
# def evaluate(sinc, spec, clf, loader, crit, device, use_amp=True):
#     """Evaluates the fusion model on a given dataset loader."""
#     sinc.eval()
#     spec.eval()
#     clf.eval()
#     total_loss = 0.0
#     all_scores = []
#     all_labels = []
#     amp_enabled = use_amp and (device.type == 'cuda')

#     with torch.no_grad(): # Essential for evaluation
#         for w1, w2, i1, i2, y in loader:
#             w1, w2, i1, i2, y = (
#                 w1.to(device), w2.to(device),
#                 i1.to(device), i2.to(device),
#                 y.float().to(device) # Ensure labels are float for BCELoss
#             )

#             # Use autocast during evaluation forward pass for consistency & speed
#             with torch.cuda.amp.autocast(enabled=amp_enabled):
#                 a1, a2 = sinc(w1), sinc(w2)
#                 s1, s2 = spec(i1), spec(i2)
#                 pred = clf(a1, a2, s1, s2)
#                 loss = crit(pred, y)

#             total_loss += loss.item() * y.size(0)
#             all_scores.append(pred.cpu())
#             all_labels.append(y.cpu())

#     avg_loss = total_loss / len(loader.dataset)
#     y_true = torch.cat(all_labels).numpy()
#     y_score = torch.cat(all_scores).numpy()

#     try:
#         auc = roc_auc_score(y_true, y_score)
#     except ValueError as e:
#         print(f"Warning: Could not compute AUC (possibly only one class in batch?): {e}")
#         auc = 0.0 # Assign a default value or handle as needed

#     return avg_loss, auc

# # --- Main Training Function ---
# def main():
#     p = argparse.ArgumentParser(description="Train Fusion Speaker Verification Model with Optimizations")
#     # Input Data Arguments
#     p.add_argument("raw_csv", help="Path to training CSV for raw audio pairs (e.g., pairs_raw_train_local.csv)")
#     p.add_argument("spec_csv", help="Path to training CSV for spectrogram pairs (e.g., pairs_spec_train_local.csv)")
#     p.add_argument("--spec_data_root", type=str, default=None,
#                   help="Local root directory for spectrogram images (if paths in spec_csv are relative or need overriding). Not needed if spec_csv contains absolute local paths.")

#     # Training Hyperparameters
#     p.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
#     p.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
#     p.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")

#     # Validation and Early Stopping
#     p.add_argument("--val_split", type=float, default=0.1, help="Fraction of data for validation set.")
#     p.add_argument("--patience", type=int, default=5, help="Patience for early stopping (epochs).")

#     # Resuming and Saving
#     p.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint.")
#     p.add_argument("--output_dir", type=Path, default=Path("outputs"), help="Directory to save checkpoints and metrics.")

#     # Technical Configuration
#     p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
#     p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for training.")
#     p.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
#     p.add_argument("--no_pin_memory", action="store_true", help="Disable pin_memory for DataLoader (use if issues occur).")
#     p.add_argument("--no_compile", action="store_true", help="Disable torch.compile optimization.")
#     p.add_argument("--no_amp", action="store_true", help="Disable Automatic Mixed Precision (AMP).")

#     args = p.parse_args()

#     # --- Setup ---
#     torch.manual_seed(args.seed)
#     if args.device == "cuda" and not torch.cuda.is_available():
#         print("Warning: CUDA requested but not available. Switching to CPU.")
#         args.device = "cpu"
#     device = torch.device(args.device)
#     pin_memory = not args.no_pin_memory and (device.type == 'cuda') # Pin memory only works well with CUDA
#     use_compile = COMPILE_SUPPORTED and not args.no_compile
#     use_amp = not args.no_amp and (device.type == 'cuda') # AMP only works on CUDA

#     print("--- Configuration ---")
#     print(f"Device: {device}")
#     print(f"Use torch.compile: {use_compile}")
#     print(f"Use Automatic Mixed Precision (AMP): {use_amp}")
#     print(f"DataLoader Workers: {args.num_workers}")
#     print(f"DataLoader Pin Memory: {pin_memory}")
#     print(f"Output Directory: {args.output_dir}")
#     print(f"Raw CSV: {args.raw_csv}")
#     print(f"Spectrogram CSV: {args.spec_csv}")
#     if args.spec_data_root:
#         print(f"Spectrogram Root Override: {args.spec_data_root}")
#     print("---------------------")


#     # --- Load Dataset ---
#     print("Loading Fusion dataset...")
#     try:
#         full_ds = FusionDataset(args.raw_csv, args.spec_csv, args.spec_data_root)
#         if len(full_ds) == 0:
#              print("Error: Dataset is empty after initialization. Exiting.")
#              return
#     except FileNotFoundError as e:
#          print(f"Error: Could not find input CSV file: {e}. Exiting.")
#          return
#     except Exception as e:
#          print(f"Error loading dataset: {e}. Exiting.")
#          return


#     # --- Train/Val Split ---
#     n_val = int(len(full_ds) * args.val_split)
#     n_train = len(full_ds) - n_val
#     if n_train <= 0 or n_val <= 0:
#          print(f"Error: Invalid train/validation split from dataset size {len(full_ds)} and split ratio {args.val_split}. Need positive samples in both.")
#          return
#     print(f"Splitting dataset: {n_train} train, {n_val} validation samples.")
#     train_ds, val_ds = random_split(
#         full_ds, [n_train, n_val],
#         generator=torch.Generator().manual_seed(args.seed)
#     )
#     train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
#                               num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=(args.num_workers > 0))
#     val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
#                             num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=(args.num_workers > 0))
#     print("DataLoaders created.")

#     # --- Initialize Models ---
#     print("Initializing models...")
#     sinc = SincNetEmbedding().to(device)
#     spec = MobileNetV2Embedding().to(device) # Assumes emb_dim=256 matches SincNet
#     clf = FusionClassifier().to(device) # Assumes emb_dim=256

#     # --- Apply torch.compile (if enabled and supported) ---
#     if use_compile:
#         print("Applying torch.compile to models...")
#         compile_start_time = time.time()
#         try:
#             sinc = torch.compile(sinc)
#             print(" - SincNet compiled successfully.")
#         except Exception as e:
#             print(f" - Warning: torch.compile failed for SincNet: {e}. Proceeding without compiling SincNet.")
#         try:
#             spec = torch.compile(spec)
#             print(" - MobileNetV2 compiled successfully.")
#         except Exception as e:
#             print(f" - Warning: torch.compile failed for MobileNetV2: {e}. Proceeding without compiling MobileNetV2.")
#         try:
#             clf = torch.compile(clf)
#             print(" - FusionClassifier compiled successfully.")
#         except Exception as e:
#             print(f" - Warning: torch.compile failed for FusionClassifier: {e}. Proceeding without compiling classifier.")
#         compile_end_time = time.time()
#         print(f"Model compilation took {compile_end_time - compile_start_time:.2f} seconds.")


#     # --- Optimizer and Criterion ---
#     # Combine parameters from all models that require gradients
#     params_to_optimize = list(sinc.parameters()) + list(spec.parameters()) + list(clf.parameters())
#     opt = torch.optim.Adam(params_to_optimize, lr=args.lr)
#     crit = nn.BCELoss() # Requires sigmoid output from the model
#     print("Optimizer and Criterion initialized.")

#     # --- AMP GradScaler (initialize even if use_amp is false, enabled=False handles it) ---
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

#     last_ckpt = ckpt_dir / "fusion_last.pt"
#     best_ckpt = ckpt_dir / "fusion_best.pt"
#     stats_fn = metr_dir / "fusion_train_stats.json"

#     # --- Resume Logic ---
#     if args.resume and last_ckpt.exists():
#         print(f"Attempting to resume training from {last_ckpt}")
#         try:
#             # Load checkpoint onto the correct device
#             # Note: If compiling, loading state dict into a compiled model might need care,
#             # but usually works if the underlying architecture hasn't changed.
#             # It's often safer to load state dict *before* compiling if resuming.
#             # However, we compile first here, so let's try loading into compiled models.
#             ck = torch.load(last_ckpt, map_location=device)

#             # Handle potential DataParallel wrapping if saved that way
#             def load_state_dict_flexible(model, state_dict):
#                  # Check if state_dict keys start with 'module.' (from DataParallel)
#                  if list(state_dict.keys())[0].startswith('module.'):
#                      # Create new state_dict without 'module.' prefix
#                      from collections import OrderedDict
#                      new_state_dict = OrderedDict()
#                      for k, v in state_dict.items():
#                          name = k[7:] # remove `module.`
#                          new_state_dict[name] = v
#                      model.load_state_dict(new_state_dict)
#                  else:
#                      model.load_state_dict(state_dict)

#             load_state_dict_flexible(sinc, ck["sinc"])
#             load_state_dict_flexible(spec, ck["spec"])
#             load_state_dict_flexible(clf, ck["clf"])

#             # Load optimizer state - this should generally work fine
#             opt.load_state_dict(ck["optimizer"])

#             # Load AMP scaler state if it exists in checkpoint
#             if "scaler" in ck and use_amp:
#                  scaler.load_state_dict(ck["scaler"])
#                  print(" - Loaded AMP scaler state.")

#             start_ep = ck.get("epoch", 1) + 1 # Use .get for safety
#             best_val_loss = ck.get("best_val_loss", float("inf"))
#             patience_ctr = ck.get("patience_ctr", 0)

#             # Load history
#             if stats_fn.exists():
#                 try:
#                     with open(stats_fn, 'r') as f:
#                         history = json.load(f)
#                     if not isinstance(history, list): history = [] # Reset if corrupt
#                     # Keep only history before the resumed epoch
#                     history = [h for h in history if h.get('epoch', 0) < start_ep]
#                     print(f" - Loaded training history up to epoch {start_ep - 1}.")
#                 except json.JSONDecodeError:
#                     print(f" - Warning: Could not parse history file {stats_fn}. Starting fresh history.")
#                     history = []
#             print(f"Resuming training from epoch {start_ep}. Best validation loss so far: {best_val_loss:.4f}")

#         except Exception as e:
#             print(f"Error loading checkpoint: {e}. Starting training from scratch.")
#             start_ep = 1
#             best_val_loss = float("inf")
#             patience_ctr = 0
#             history = []

#     # --- Training Loop ---
#     print(f"\nStarting Fusion model training from epoch {start_ep} to {args.epochs}...")
#     training_start_time = time.time()

#     for epoch in range(start_ep, args.epochs + 1):
#         epoch_start_time = time.time()
#         # --- Train ---
#         sinc.train()
#         spec.train()
#         clf.train()
#         train_losses = []
#         print(f"\n--- Epoch {epoch}/{args.epochs} ---")
#         print("Training...")
#         batch_times = []
#         for i, (w1, w2, i1, i2, y) in enumerate(train_loader):
#             batch_start_time = time.time()
#             w1, w2, i1, i2, y = (
#                 w1.to(device, non_blocking=pin_memory),
#                 w2.to(device, non_blocking=pin_memory),
#                 i1.to(device, non_blocking=pin_memory),
#                 i2.to(device, non_blocking=pin_memory),
#                 y.float().to(device, non_blocking=pin_memory) # Ensure labels are float
#             )

#             opt.zero_grad(set_to_none=True) # More efficient zeroing

#             # Forward pass with AMP autocast
#             with torch.cuda.amp.autocast(enabled=use_amp):
#                 a1, a2 = sinc(w1), sinc(w2)
#                 s1, s2 = spec(i1), spec(i2)
#                 pred = clf(a1, a2, s1, s2)
#                 loss = crit(pred, y)

#             # Scaled backward pass
#             scaler.scale(loss).backward()

#             # Optimizer step (with gradient unscaling)
#             scaler.step(opt)

#             # Update scaler
#             scaler.update()

#             train_losses.append(loss.item() * y.size(0))
#             batch_end_time = time.time()
#             batch_times.append(batch_end_time - batch_start_time)

#             # Optional batch progress print
#             if (i + 1) % 50 == 0 or i == len(train_loader) - 1:
#                  avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
#                  print(f"  Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Avg Batch Time: {avg_batch_time:.3f}s")
#                  batch_times = [] # Reset for next interval


#         train_loss = sum(train_losses) / len(train_ds)
#         print(f"Epoch {epoch} Training complete. Average Train Loss: {train_loss:.4f}")

#         # --- Validate ---
#         print("Validating...")
#         val_start_time = time.time()
#         val_loss, val_auc = evaluate(sinc, spec, clf, val_loader, crit, device, use_amp)
#         val_end_time = time.time()
#         print(f"Epoch {epoch} Validation complete. Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f} (Time: {val_end_time - val_start_time:.2f}s)")

#         # --- Log History ---
#         history.append({
#             "epoch": epoch,
#             "train_loss": train_loss,
#             "val_loss": val_loss,
#             "val_auc": val_auc
#         })
#         try:
#             with open(stats_fn, "w") as f:
#                 json.dump(history, f, indent=2)
#         except Exception as e:
#             print(f"Warning: Could not save training history to {stats_fn}: {e}")

#         # --- Checkpointing and Early Stopping ---
#         print("Saving last checkpoint...")
#         checkpoint = {
#             "epoch": epoch,
#             # Save state dicts - handle potential compiled model wrappers if necessary
#             # .state_dict() should work correctly even on compiled models
#             "sinc": sinc.state_dict(),
#             "spec": spec.state_dict(),
#             "clf": clf.state_dict(),
#             "optimizer": opt.state_dict(),
#             "best_val_loss": best_val_loss, # Save the best loss *before* this epoch's check
#             "patience_ctr": patience_ctr,
#         }
#         if use_amp:
#              checkpoint["scaler"] = scaler.state_dict() # Save scaler state if using AMP

#         try:
#             torch.save(checkpoint, last_ckpt)
#             # print(f"  Saved last checkpoint to {last_ckpt}") # Can be verbose
#         except Exception as e:
#             print(f"Warning: Failed to save last checkpoint {last_ckpt}: {e}")


#         print("Checking for improvement...")
#         if val_loss < best_val_loss:
#             print(f"  Validation loss improved ({best_val_loss:.4f} --> {val_loss:.4f}). Saving best model.")
#             best_val_loss = val_loss
#             patience_ctr = 0
#             # Save best checkpoint (potentially smaller, without optimizer/scaler)
#             best_checkpoint = {
#                 "sinc": sinc.state_dict(),
#                 "spec": spec.state_dict(),
#                 "clf": clf.state_dict(),
#                 "best_val_loss": best_val_loss,
#                 "epoch": epoch # Record epoch when best was achieved
#             }
#             try:
#                 torch.save(best_checkpoint, best_ckpt)
#                 print(f"  Saved best checkpoint to {best_ckpt}")
#             except Exception as e:
#                  print(f"Warning: Failed to save best checkpoint {best_ckpt}: {e}")
#         else:
#             patience_ctr += 1
#             print(f"  Validation loss did not improve. Patience: {patience_ctr}/{args.patience}")
#             if patience_ctr >= args.patience:
#                 print(f"Early stopping triggered at epoch {epoch} due to lack of improvement.")
#                 break # Exit training loop

#         epoch_end_time = time.time()
#         print(f"Epoch {epoch} finished in {epoch_end_time - epoch_start_time:.2f} seconds.")


#     # --- End of Training ---
#     training_end_time = time.time()
#     print("\nFusion model training finished.")
#     print(f"Total training time: {training_end_time - training_start_time:.2f} seconds.")
#     print(f"Final training history saved to {stats_fn}")
#     print(f"Last checkpoint saved to {last_ckpt}")
#     print(f"Best checkpoint saved to {best_ckpt} (Val Loss: {best_val_loss:.4f})")

# if __name__ == "__main__":
#     main()

# # # scripts/train_fusion.py

# # import argparse, json, os
# # from pathlib import Path

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch.utils.data import DataLoader, random_split
# # from sklearn.metrics import roc_auc_score

# # from datasets.raw_dataset import RawAudioPairDataset # Make sure this path is correct
# # from datasets.spectrogram_dataset import MelSpectrogramPairDataset # Make sure this path is correct
# # from models.sincnet import SincNetEmbedding # Make sure this path is correct
# # from models.mobilenet_embedding import MobileNetV2Embedding # Make sure this path is correct

# # class FusionDataset(torch.utils.data.Dataset):
# #     def __init__(self, raw_csv, spec_csv, spec_root=None):
# #         print(f"Initializing FusionDataset: Raw CSV='{raw_csv}', Spec CSV='{spec_csv}'")
# #         self.raw  = RawAudioPairDataset(raw_csv)
# #         self.spec = MelSpectrogramPairDataset(spec_csv)
# #         if spec_root:
# #             print(f"Remapping spectrogram paths to root: {spec_root}")
# #             # remap the .pairs in spec
# #             allp = self.spec.pairs
# #             if not allp:
# #                  print("Warning: Spectrogram dataset pairs list is empty before remapping.")
# #                  self.spec.pairs = []
# #                  return

# #             try:
# #                 orig_root = os.path.commonpath([p for p,_,_ in allp] + [q for _,q,_ in allp])
# #                 print(f"Inferred original common path for spectrograms: {orig_root}")
# #             except ValueError:
# #                  print("Warning: Could not determine common path for spectrograms. Assuming relative paths.")
# #                  orig_root = None # Handle cases where paths might not have a common root

# #             new = []
# #             skipped_count = 0
# #             for p1,p2,l in allp:
# #                  try:
# #                      if orig_root:
# #                          rel1 = os.path.relpath(p1, orig_root)
# #                          rel2 = os.path.relpath(p2, orig_root)
# #                      else: # Assume paths are already relative or absolute as needed
# #                          rel1 = p1
# #                          rel2 = p2
# #                      new.append((os.path.join(spec_root, rel1),
# #                                  os.path.join(spec_root, rel2),
# #                                  l))
# #                  except ValueError as e:
# #                     # This might happen if paths are on different drives (Windows) or structured unexpectedly
# #                     print(f"Skipping pair due to path issue: ({p1}, {p2}). Error: {e}")
# #                     skipped_count += 1
# #                     continue

# #             if skipped_count > 0:
# #                 print(f"Warning: Skipped {skipped_count} pairs due to path issues during remapping.")
# #             self.spec.pairs = new
# #             print(f"Remapping complete. Number of spectrogram pairs: {len(self.spec.pairs)}")

# #         if len(self.raw) != len(self.spec):
# #              print(f"CRITICAL WARNING: Mismatch in dataset lengths after loading/remapping! Raw: {len(self.raw)}, Spec: {len(self.spec)}. Fusion will likely fail.")
# #         else:
# #              print(f"Raw and Spectrogram datasets loaded successfully. Length: {len(self.raw)}")

# #     def __len__(self): return len(self.raw) # Use raw length, assuming they should match
# #     def __getitem__(self, idx):
# #         w1,w2,y = self.raw[idx]
# #         # Add a check in case spec length is mismatched (though it shouldn't be)
# #         if idx < len(self.spec):
# #              i1,i2,_= self.spec[idx]
# #         else:
# #              # Handle error case: return dummy tensors or raise an error
# #              print(f"Error: Index {idx} out of bounds for spectrogram dataset (len {len(self.spec)}). Returning dummy data.")
# #              # Example dummy data (adjust size as needed):
# #              i1 = torch.zeros(3, 224, 224) # Assuming spec images are 3x224x224
# #              i2 = torch.zeros(3, 224, 224)
# #              # Or raise IndexError(f"Index {idx} out of bounds for spectrogram dataset")
# #         return w1,w2,i1,i2,y

# # class FusionClassifier(nn.Module):
# #     def __init__(self, emb_dim=256, hidden_dim=512):
# #         super().__init__()
# #         in_dim = 2*2*emb_dim # (a1, s1) + (a2, s2)
# #         self.fc1 = nn.Linear(in_dim, hidden_dim)
# #         self.bn1 = nn.BatchNorm1d(hidden_dim)
# #         self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
# #         self.bn2 = nn.BatchNorm1d(hidden_dim//2)
# #         self.out = nn.Linear(hidden_dim//2, 1)
# #     def forward(self,a1,a2,s1,s2):
# #         # a1, a2 are SincNet embeddings [B, emb_dim]
# #         # s1, s2 are Spec embeddings [B, emb_dim]
# #         e1 = torch.cat([a1,s1],dim=1) # [B, 2*emb_dim]
# #         e2 = torch.cat([a2,s2],dim=1) # [B, 2*emb_dim]
# #         x  = torch.cat([e1,e2],dim=1) # [B, 4*emb_dim] == [B, in_dim]
# #         x  = F.relu(self.bn1(self.fc1(x)))
# #         x  = F.relu(self.bn2(self.fc2(x)))
# #         return torch.sigmoid(self.out(x)).squeeze(1)

# # def evaluate(sinc, spec, clf, loader, crit, device):
# #     sinc.eval(); spec.eval(); clf.eval()
# #     losses, scores, labels = [], [], []
# #     with torch.no_grad():
# #         for w1,w2,i1,i2,y in loader:
# #             w1,w2,i1,i2,y = (
# #                 w1.to(device), w2.to(device),
# #                 i1.to(device), i2.to(device),
# #                 y.float().to(device)
# #             )
# #             a1,a2 = sinc(w1), sinc(w2)
# #             s1,s2 = spec(i1), spec(i2)
# #             pred   = clf(a1,a2,s1,s2)
# #             losses.append(crit(pred,y).item()*y.size(0))
# #             scores.append(pred.cpu()); labels.append(y.cpu())
# #     avg_loss = sum(losses)/len(loader.dataset)
# #     y_true = torch.cat(labels).numpy()
# #     y_score= torch.cat(scores).numpy()
# #     auc = roc_auc_score(y_true, y_score)
# #     return avg_loss, auc

# # def main():
# #     p = argparse.ArgumentParser()
# #     p.add_argument("raw_csv",  help="pairs_raw_train.csv")
# #     p.add_argument("spec_csv", help="pairs_spec_train.csv")
# #     p.add_argument("--spec_data_root", type=str, default=None,
# #                    help="Local root for spectrogram images")
# #     p.add_argument("--epochs",     type=int,   default=50)
# #     p.add_argument("--batch_size", type=int,   default=32)
# #     p.add_argument("--lr",         type=float, default=1e-4)
# #     p.add_argument("--val_split",  type=float, default=0.1)
# #     p.add_argument("--patience",   type=int,   default=5)
# #     p.add_argument("--resume",     action="store_true")
# #     p.add_argument("--seed",       type=int,   default=42)
# #     p.add_argument("--output_dir", type=Path,  default=Path("outputs"))
# #     p.add_argument("--device",     type=str,   default="cuda")
# #     args = p.parse_args()

# #     torch.manual_seed(args.seed)
# #     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
# #     print(f"Using device: {device}")

# #     # load & remap dataset
# #     print("Loading Fusion dataset...")
# #     full_ds = FusionDataset(args.raw_csv, args.spec_csv, args.spec_data_root)
# #     if len(full_ds.raw) != len(full_ds.spec):
# #          print("Error: Dataset length mismatch detected after initialization. Cannot proceed.")
# #          return
# #     if len(full_ds) == 0:
# #         print("Error: Fusion dataset is empty. Check CSV paths and spec_data_root. Exiting.")
# #         return

# #     n_val   = int(len(full_ds) * args.val_split)
# #     n_train = len(full_ds) - n_val
# #     print(f"Split: {n_train} train, {n_val} validation samples.")
# #     train_ds, val_ds = random_split(
# #         full_ds, [n_train, n_val],
# #         generator=torch.Generator().manual_seed(args.seed)
# #     )
# #     train_loader = DataLoader(train_ds, batch_size=args.batch_size,
# #                               shuffle=True, num_workers=4, pin_memory=True)
# #     val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
# #                               shuffle=False,num_workers=4,pin_memory=True)
# #     print("DataLoaders created.")

# #     # Models, optimizer, criterion
# #     print("Initializing SincNet, MobileNetV2, and FusionClassifier models...")
# #     sinc = SincNetEmbedding().to(device)
# #     spec = MobileNetV2Embedding().to(device)
# #     clf  = FusionClassifier().to(device)
# #     opt  = torch.optim.Adam(
# #         list(sinc.parameters())+list(spec.parameters())+list(clf.parameters()),
# #         lr=args.lr
# #     )
# #     crit = nn.BCELoss()
# #     print("Models, Optimizer, and Criterion initialized.")

# #     ckpt_dir = args.output_dir/"checkpoints"; ckpt_dir.mkdir(parents=True,exist_ok=True)
# #     metr_dir = args.output_dir/"metrics";    metr_dir.mkdir(parents=True,exist_ok=True)
# #     print(f"Output directories prepared: {ckpt_dir}, {metr_dir}")


# #     start_ep, best_val, patience = 1, float("inf"), 0
# #     history = []

# #     last_ckpt = ckpt_dir/"fusion_last.pt"
# #     best_ckpt = ckpt_dir/"fusion_best.pt"
# #     stats_fn  = metr_dir/"fusion_train_stats.json"

# #     if args.resume and last_ckpt.exists():
# #         print(f"Attempting to resume training from {last_ckpt}")
# #         ck = torch.load(last_ckpt, map_location=device)
# #         sinc.load_state_dict(ck["sinc"])
# #         spec.load_state_dict(ck["spec"])
# #         clf.load_state_dict(ck["clf"])
# #         opt.load_state_dict(ck["optimizer"])
# #         start_ep    = ck["epoch"] + 1
# #         best_val    = ck["best_val_loss"]
# #         patience    = ck["patience_ctr"]
# #         if stats_fn.exists():
# #              try:
# #                  with open(stats_fn, 'r') as f:
# #                     history = json.load(f)
# #                  # Ensure history is a list
# #                  if not isinstance(history, list):
# #                     print(f"Warning: Corrupted history file {stats_fn}. Starting fresh history.")
# #                     history = []
# #                  else:
# #                     # Optional: Trim history if it goes beyond the resumed epoch
# #                     history = [h for h in history if h.get('epoch', 0) < start_ep]
# #              except json.JSONDecodeError:
# #                 print(f"Warning: Could not parse {stats_fn}. Starting fresh history.")
# #                 history = []
# #         print(f"Resuming Fusion model training from epoch {start_ep}")

# #     print(f"\nStarting Fusion model training from epoch {start_ep} to {args.epochs}...")
# #     for epoch in range(start_ep, args.epochs+1):
# #         sinc.train(); spec.train(); clf.train()
# #         train_losses = []
# #         print(f"\n--- Epoch {epoch}/{args.epochs} ---")
# #         print("Training Fusion model...")
# #         for i, (w1,w2,i1,i2,y) in enumerate(train_loader):
# #             w1,w2,i1,i2,y = (
# #                 w1.to(device),w2.to(device),
# #                 i1.to(device),i2.to(device),
# #                 y.float().to(device)
# #             )
# #             a1,a2 = sinc(w1), sinc(w2)
# #             s1,s2 = spec(i1), spec(i2)
# #             pred   = clf(a1,a2,s1,s2)
# #             loss   = crit(pred,y)
# #             opt.zero_grad(); loss.backward(); opt.step()
# #             train_losses.append(loss.item()*y.size(0))
# #             # Optional batch progress print
# #             # if (i + 1) % 50 == 0:
# #             #      print(f"  Batch {i+1}/{len(train_loader)}, Current Batch Loss: {loss.item():.4f}")

# #         train_loss = sum(train_losses)/len(train_ds)
# #         print(f"Epoch {epoch} Training complete. Average Train Loss: {train_loss:.4f}")


# #         print("Validating Fusion model...")
# #         val_loss, val_auc = evaluate(sinc, spec, clf, val_loader, crit, device)
# #         print(f"Epoch {epoch} Validation complete. Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

# #         history.append({"epoch":epoch,"train_loss":train_loss,
# #                         "val_loss":val_loss,"val_auc":val_auc})
# #         try:
# #             with open(stats_fn,"w") as f: json.dump(history,f,indent=2)
# #         except Exception as e:
# #             print(f"Error saving stats file: {e}")


# #         print("Saving checkpoint...")
# #         # Always save last checkpoint
# #         try:
# #             torch.save({
# #                 "epoch":         epoch,
# #                 "sinc":          sinc.state_dict(),
# #                 "spec":          spec.state_dict(),
# #                 "clf":           clf.state_dict(),
# #                 "optimizer":     opt.state_dict(),
# #                 "best_val_loss": best_val,
# #                 "patience_ctr":  patience
# #             }, last_ckpt)
# #             print(f"  Saved last checkpoint to {last_ckpt}")
# #         except Exception as e:
# #              print(f"Error saving last checkpoint: {e}")

# #         # Original print statement (optional)
# #         # print(f"[Fusion] Ep{epoch}: train={train_loss:.4f} val={val_loss:.4f} auc={val_auc:.4f}")

# #         # Check for improvement and save best checkpoint
# #         print("Checking for improvement...")
# #         if val_loss < best_val:
# #             print(f"  Validation loss improved ({best_val:.4f} --> {val_loss:.4f}). Saving best model.")
# #             best_val = val_loss; patience = 0
# #             try:
# #                 torch.save({
# #                     "sinc":          sinc.state_dict(),
# #                     "spec":          spec.state_dict(),
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

# #     print("\nFusion model training finished.")
# #     print(f"Final training history saved to {stats_fn}")
# #     print(f"Last checkpoint saved to {last_ckpt}")
# #     print(f"Best checkpoint saved to {best_ckpt}")


# # if __name__=="__main__":
# #     main()
# # # # scripts/train_fusion.py

# # # import argparse, json, os
# # # from pathlib import Path

# # # import torch
# # # import torch.nn as nn
# # # import torch.nn.functional as F
# # # from torch.utils.data import DataLoader, random_split
# # # from sklearn.metrics import roc_auc_score

# # # from datasets.raw_dataset import RawAudioPairDataset
# # # from datasets.spectrogram_dataset import MelSpectrogramPairDataset
# # # from models.sincnet import SincNetEmbedding
# # # from models.mobilenet_embedding import MobileNetV2Embedding

# # # class FusionDataset(torch.utils.data.Dataset):
# # #     def __init__(self, raw_csv, spec_csv, spec_root=None):
# # #         self.raw  = RawAudioPairDataset(raw_csv)
# # #         self.spec = MelSpectrogramPairDataset(spec_csv)
# # #         if spec_root:
# # #             # remap the .pairs in spec
# # #             allp = self.spec.pairs
# # #             orig_root = os.path.commonpath([p for p,_,_ in allp] + [q for _,q,_ in allp])
# # #             new = []
# # #             for p1,p2,l in allp:
# # #                 rel1 = os.path.relpath(p1, orig_root)
# # #                 rel2 = os.path.relpath(p2, orig_root)
# # #                 new.append((os.path.join(spec_root, rel1),
# # #                             os.path.join(spec_root, rel2),
# # #                             l))
# # #             self.spec.pairs = new
# # #         assert len(self.raw)==len(self.spec)
# # #     def __len__(self): return len(self.raw)
# # #     def __getitem__(self, idx):
# # #         w1,w2,y = self.raw[idx]
# # #         i1,i2,_= self.spec[idx]
# # #         return w1,w2,i1,i2,y

# # # class FusionClassifier(nn.Module):
# # #     def __init__(self, emb_dim=256, hidden_dim=512):
# # #         super().__init__()
# # #         in_dim = 2*2*emb_dim
# # #         self.fc1 = nn.Linear(in_dim, hidden_dim)
# # #         self.bn1 = nn.BatchNorm1d(hidden_dim)
# # #         self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
# # #         self.bn2 = nn.BatchNorm1d(hidden_dim//2)
# # #         self.out = nn.Linear(hidden_dim//2, 1)
# # #     def forward(self,a1,a2,s1,s2):
# # #         e1 = torch.cat([a1,s1],dim=1)
# # #         e2 = torch.cat([a2,s2],dim=1)
# # #         x  = torch.cat([e1,e2],dim=1)
# # #         x  = F.relu(self.bn1(self.fc1(x)))
# # #         x  = F.relu(self.bn2(self.fc2(x)))
# # #         return torch.sigmoid(self.out(x)).squeeze(1)

# # # def evaluate(sinc, spec, clf, loader, crit, device):
# # #     sinc.eval(); spec.eval(); clf.eval()
# # #     losses, scores, labels = [], [], []
# # #     with torch.no_grad():
# # #         for w1,w2,i1,i2,y in loader:
# # #             w1,w2,i1,i2,y = (
# # #                 w1.to(device), w2.to(device),
# # #                 i1.to(device), i2.to(device),
# # #                 y.float().to(device)
# # #             )
# # #             a1,a2 = sinc(w1), sinc(w2)
# # #             s1,s2 = spec(i1), spec(i2)
# # #             pred   = clf(a1,a2,s1,s2)
# # #             losses.append(crit(pred,y).item()*y.size(0))
# # #             scores.append(pred.cpu()); labels.append(y.cpu())
# # #     avg_loss = sum(losses)/len(loader.dataset)
# # #     y_true = torch.cat(labels).numpy()
# # #     y_score= torch.cat(scores).numpy()
# # #     auc = roc_auc_score(y_true, y_score)
# # #     return avg_loss, auc

# # # def main():
# # #     p = argparse.ArgumentParser()
# # #     p.add_argument("raw_csv",  help="pairs_raw_train.csv")
# # #     p.add_argument("spec_csv", help="pairs_spec_train.csv")
# # #     p.add_argument("--spec_data_root", type=str, default=None,
# # #                    help="Local root for .pngs")
# # #     p.add_argument("--epochs",     type=int,   default=50)
# # #     p.add_argument("--batch_size", type=int,   default=32)
# # #     p.add_argument("--lr",         type=float, default=1e-4)
# # #     p.add_argument("--val_split",  type=float, default=0.1)
# # #     p.add_argument("--patience",   type=int,   default=5)
# # #     p.add_argument("--resume",     action="store_true")
# # #     p.add_argument("--seed",       type=int,   default=42)
# # #     p.add_argument("--output_dir", type=Path,  default=Path("outputs"))
# # #     p.add_argument("--device",     type=str,   default="cuda")
# # #     args = p.parse_args()

# # #     torch.manual_seed(args.seed)
# # #     device = torch.device(args.device if torch.cuda.is_available() else "cpu")

# # #     # load & remap
# # #     full_ds = FusionDataset(args.raw_csv, args.spec_csv, args.spec_data_root)
# # #     n_val   = int(len(full_ds) * args.val_split)
# # #     train_ds, val_ds = random_split(
# # #         full_ds, [len(full_ds)-n_val, n_val],
# # #         generator=torch.Generator().manual_seed(args.seed)
# # #     )
# # #     train_loader = DataLoader(train_ds, batch_size=args.batch_size,
# # #                               shuffle=True, num_workers=4, pin_memory=True)
# # #     val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
# # #                               shuffle=False,num_workers=4,pin_memory=True)

# # #     sinc = SincNetEmbedding().to(device)
# # #     spec = MobileNetV2Embedding().to(device)
# # #     clf  = FusionClassifier().to(device)
# # #     opt  = torch.optim.Adam(
# # #         list(sinc.parameters())+list(spec.parameters())+list(clf.parameters()),
# # #         lr=args.lr
# # #     )
# # #     crit = nn.BCELoss()

# # #     ckpt_dir = args.output_dir/"checkpoints"; ckpt_dir.mkdir(parents=True,exist_ok=True)
# # #     metr_dir = args.output_dir/"metrics";    metr_dir.mkdir(parents=True,exist_ok=True)

# # #     start_ep, best_val, patience = 1, float("inf"), 0
# # #     history = []

# # #     last_ckpt = ckpt_dir/"fusion_last.pt"
# # #     best_ckpt = ckpt_dir/"fusion_best.pt"
# # #     stats_fn  = metr_dir/"fusion_train_stats.json"

# # #     if args.resume and last_ckpt.exists():
# # #         ck = torch.load(last_ckpt, map_location=device)
# # #         sinc.load_state_dict(ck["sinc"])
# # #         spec.load_state_dict(ck["spec"])
# # #         clf.load_state_dict(ck["clf"])
# # #         opt.load_state_dict(ck["optimizer"])
# # #         start_ep    = ck["epoch"] + 1
# # #         best_val    = ck["best_val_loss"]
# # #         patience    = ck["patience_ctr"]
# # #         if stats_fn.exists():
# # #             history = json.load(stats_fn)
# # #         print(f"Resuming from epoch {start_ep}")

# # #     for epoch in range(start_ep, args.epochs+1):
# # #         sinc.train(); spec.train(); clf.train()
# # #         train_losses = []
# # #         for w1,w2,i1,i2,y in train_loader:
# # #             w1,w2,i1,i2,y = (
# # #                 w1.to(device),w2.to(device),
# # #                 i1.to(device),i2.to(device),
# # #                 y.float().to(device)
# # #             )
# # #             a1,a2 = sinc(w1), sinc(w2)
# # #             s1,s2 = spec(i1), spec(i2)
# # #             pred   = clf(a1,a2,s1,s2)
# # #             loss   = crit(pred,y)
# # #             opt.zero_grad(); loss.backward(); opt.step()
# # #             train_losses.append(loss.item()*y.size(0))

# # #         train_loss = sum(train_losses)/len(train_ds)
# # #         val_loss, val_auc = evaluate(sinc, spec, clf, val_loader, crit, device)
# # #         history.append({"epoch":epoch,"train_loss":train_loss,
# # #                         "val_loss":val_loss,"val_auc":val_auc})
# # #         with open(stats_fn,"w") as f: json.dump(history,f,indent=2)

# # #         torch.save({
# # #             "epoch":         epoch,
# # #             "sinc":          sinc.state_dict(),
# # #             "spec":          spec.state_dict(),
# # #             "clf":           clf.state_dict(),
# # #             "optimizer":     opt.state_dict(),
# # #             "best_val_loss": best_val,
# # #             "patience_ctr":  patience
# # #         }, last_ckpt)

# # #         print(f"[Fusion] Ep{epoch}: train={train_loss:.4f} val={val_loss:.4f} auc={val_auc:.4f}")

# # #         if val_loss < best_val:
# # #             best_val = val_loss; patience = 0
# # #             torch.save({
# # #                 "sinc":          sinc.state_dict(),
# # #                 "spec":          spec.state_dict(),
# # #                 "clf":           clf.state_dict(),
# # #                 "optimizer":     opt.state_dict(),
# # #                 "best_val_loss": best_val,
# # #                 "patience_ctr":  patience
# # #             }, best_ckpt)
# # #         else:
# # #             patience += 1
# # #             if patience >= args.patience:
# # #                 print(f"Early stopping at epoch {epoch}")
# # #                 break

# # # if __name__=="__main__":
# # #     main()