# Speaker Recognition and Authentication System (SincNet + Mel-CNN Fusion)

This project implements a complete speaker identification and authentication pipeline using:

- **SincNet**: A raw-audio-based deep learning model for speaker embeddings.
- **Mel CNN**: A CNN trained on Mel spectrograms.
- **Fusion Model**: Attention-based feature-level fusion of SincNet and Mel CNN embeddings with a final decision MLP.

---

## 🔧 Directory Structure

```
final_fusion/
├── cli_cnn.py               # CLI tool to tune, train, test the Mel CNN and extract embeddings
├── cli_fusion.py            # CLI tool to tune, train, test the fused model (SincNet + CNN)
├── cli_sincnet_test.py      # CLI tool to load saved SincNet embeddings and evaluate them
├── models.py                # Contains CNN, Fusion model, Attention module, Decision MLP
├── data_utils.py            # VoxCeleb loader supporting nested directory structure and Mel processing
├── plots/                   # Plots for loss/accuracy over time, comparisons etc.
├── metrics/                 # Stores CSVs with training, validation, and test metrics
├── logs/                    # Training logs
└── main.ipynb               # Orchestrates all tasks using magic commands
```

---

## 📜 How to Run

### 1. **Tune, Train, and Extract Embeddings from CNN (Mel Spectrograms)**

```bash
!python cli_cnn.py --mode tune --dataset_path /path/to/train --output_path /path/to/save
```

Other modes:
```bash
--mode train      # Train final CNN with best config on full train set
--mode test       # Evaluate CNN on test set, output metrics + predictions
```

Options:
```bash
--n_trials 5              # Number of tuning trials
--batch_size 32
--num_epochs 20
--target_frames 300       # Ensures fixed-size Mel spectrograms
```

---

### 2. **Evaluate Pretrained SincNet Embeddings**

```bash
!python cli_sincnet_test.py --embedding_dir /path/to/SincNet --test_dir /path/to/test
```

- Uses saved `.npy` or `.pkl` embeddings from SincNet.
- Evaluates on test set, provides metrics for:
  - Speaker identification
  - Speaker authentication (genuine vs imposter)

---

### 3. **Fusion Model (Attention + MLP Decision)**

```bash
!python cli_fusion.py --mode tune --sincnet_dir /path/to/SincNet --cnn_dir /path/to/CNN --output_path /path/to/save
```

Other modes:
```bash
--mode train      # Train on fused embeddings
--mode test       # Evaluate on test set (fusion vs CNN vs SincNet)
```

Options:
```bash
--embedding_dim 256       # Size of each input embedding vector
--fusion_type attention   # Feature-level attention-based fusion
```

---

## 📈 Output Artifacts

All models output:

- `metrics/` with:
  - `train_metrics.csv`
  - `val_metrics.csv`
  - `test_metrics.csv`
- `plots/` with:
  - Epoch-wise accuracy/loss plots
  - Comparison bar plots (SincNet vs CNN vs Fusion)
- `logs/` with:
  - Text logs of each stage
- `checkpoints/` for each epoch
- `best_model.pth` for each model type

---

## 📊 Visualizations & Analysis

The `main.ipynb` notebook performs:

1. CNN tuning/training/testing via magic commands
2. SincNet evaluation loading precomputed features
3. Fusion model evaluation
4. Data analysis:
   - ROC Curves for authentication
   - Accuracy and F1 comparisons
   - Speaker confusion matrices
   - Epoch-wise plots for loss/accuracy

---

## ✅ Tasks Covered

| Task | Description |
|------|-------------|
| **CNN Tuning** | Hyperparameter search with validation |
| **CNN Training** | Uses best config on full training set |
| **CNN Testing** | Evaluates model on test set |
| **SincNet Testing** | Loads SincNet embeddings + evaluates |
| **Fusion Training** | Attention + MLP on fused embeddings |
| **Fusion Testing** | Identification + authentication |
| **Metrics Saving** | Epoch-wise loss/accuracy saved to CSV |
| **Visualization** | Plots generated + saved to plots dir |

---

## 📝 Notes

- You must provide `--dataset_path` for CNN and test sets.
- You must provide `--sincnet_dir` with saved `.npy` and `.pkl` files.
- All results and checkpoints are reproducibly saved per run.
