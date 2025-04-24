# Noise-Robust Speaker Verification using Multi-Modal Fusion and Fine-tuning

## Project Overview

This project implements and evaluates a speaker verification (SV) system designed for robustness against background noise and channel effects. It utilizes a multi-modal approach, fusing features extracted from raw audio waveforms (via SincNet) and mel-spectrograms (via MobileNetV2). Initial training of the embedding extractors is performed on augmented noisy data. Subsequently, the entire network (including embedders and a fusion classifier) is fine-tuned on a mixed dataset containing both clean and noisy samples to enhance robustness. The final system is evaluated across clean, noisy, and noise-filtered test conditions using the VoxCeleb1 dataset.

## Pipeline Stages

The project follows these main stages:

1.  **Setup & Data Acquisition:** Downloading VoxCeleb1 and MUSAN datasets.
2.  **Data Augmentation & Filtering:** Creating noisy versions of the training set and noisy/filtered versions of the test set.
3.  **Pair Generation:** Creating genuine/imposter pairs for clean and noisy training sets, and for clean, noisy, and filtered test sets.
4.  **Feature Generation:** Generating mel-spectrogram images (from clean and noisy data) and pre-processing raw audio to `.npy` format (clean and noisy).
5.  **Data Management:** Copying datasets locally for efficient processing and managing persistent storage on Google Drive.
6.  **Initial Model Training:** Training individual embedding models (SincNet, MobileNetV2) primarily on *noisy* augmented data.
7.  **Fine-tuning:** Fine-tuning the complete fusion model (SincNet + MobileNetV2 + Fusion Classifier) with unfrozen embedders on a *mixed* dataset of clean and noisy training pairs.
8.  **Evaluation:** Assessing the performance of the initial individual models and the final fine-tuned fusion model across different test conditions.

## 1. Datasets Used

* **VoxCeleb1:** Publicly available dataset for speaker recognition derived from YouTube videos. Used for speaker identity information.
    * `Dev` split: Used as the basis for clean and noisy training/validation sets.
    * `Test` split: Used as the basis for clean, noisy, and filtered evaluation sets.
* **MUSAN:** Corpus of music, speech, and noise recordings. Used for data augmentation to simulate noisy environments.

## 2. Data Preparation

### 2.1. Augmentation (`pythonAugment.ipynb`)

* **Noise Addition:** Clean audio from VoxCeleb1 (`Dev` and `Test` splits) was augmented by adding random noise segments (music, ambient noise) from MUSAN at random SNRs (5-25 dB).
* **Telephone Filter:** A 4th-order Butterworth bandpass filter (300-3400 Hz) was applied to all augmented audio to simulate telephone channel effects.
* **Output:**
    * `Dev_Augmented/wav`: Noisy training set base.
    * `Test_Augmented/wav`: Noisy test set base.

### 2.2. Noise Filtering (`DeepFilter.ipynb`)

* **Tool:** DeepFilterNet2 (DF3 model via `deepfilternet` library) was used for noise reduction.
* **Input:** The augmented noisy test set (`Test_Augmented/wav`).
* **Output:** `Test_Filtered/wav`: Enhanced (noise-filtered) test set base.

### 2.3. Final Datasets for Training & Testing

* **Initial Training Data:** Primarily uses data derived from `Dev_Augmented` (Noisy Train).
* **Fine-tuning Data:** Uses a mix of data derived from the original clean `Dev` set and the `Dev_Augmented` set.
* **Test Data Conditions:**
    1.  **Clean:** Derived from original `Test` split.
    2.  **Noisy:** Derived from `Test_Augmented` split.
    3.  **Filtered:** Derived from `Test_Filtered` split.

## 3. Pair Generation & Feature Preparation

### 3.1. Pair Generation (`make_verification_pairs.py`)

* **Process:** Generates CSV files containing genuine (same speaker) and imposter (different speaker) pairs for:
    * Clean Training (`pairs_raw_clean_train.csv`)
    * Noisy Training (`pairs_raw_train.csv`)
    * Clean Test (`pairs_raw_clean_test.csv`)
    * Noisy Test (`pairs_raw_noisy_test.csv`)
    * Filtered Test (`pairs_raw_filtered_test.csv`)
* **Format:** `path/to/wav1,path/to/wav2,label` (1=genuine, 0=imposter). Paths are relative to the workspace root, pointing to the corresponding WAV files (clean, noisy, or filtered).

### 3.2. Raw Audio Preprocessing (`data_manager_v3.py`)

* **Action:** `--preprocess_raw` with appropriate `--dataset_keys`.
* **Input:** Local raw audio WAV files (clean, noisy, filtered - copied from Drive).
* **Process:**
    * Resamples audio to 16kHz (`--target_sr`).
    * Pads or truncates audio to a fixed length (e.g., 3 seconds / 48000 samples via `--target_len`).
* **Output:** Saves processed audio as `.npy` files locally (`/content/data/raw_audio_preprocessed/{condition}/...`) AND to Google Drive (`$WORKSPACE/data/raw_audio_preprocessed/{condition}/...`).

### 3.3. Spectrogram Generation (`make_spectrogram_pairs.py`)

* **Action:** Run via notebook cells (e.g., `finetune_full_prep_notebook.ipynb`).
* **Input:** Original raw pair CSVs (`pairs_raw_*.csv`) and corresponding local raw WAV files.
* **Process:**
    * Generates mel-spectrograms using high-quality parameters (e.g., `n_mels=80`, `n_fft=512`, `hop_length=160`, `duration=3.0`, `img_size=224`).
    * Saves images as PNG (`img_format=png`).
* **Output:**
    * Saves spectrogram PNG images locally (`/content/data/spectrograms/{condition}/...`).
    * Saves corresponding pair CSVs (`pairs_spec_*_local.csv`) to Drive, containing paths to the *local* PNG images.
    * Generated images are also backed up to Drive (`$WORKSPACE/data/spectrograms_generated_png/{condition}/...`).

### 3.4. NPY Pair CSV Generation (`data_manager_v3.py`)

* **Action:** `--gen_csv_local`, `--gen_csv_drive` with appropriate `--dataset_keys`.
* **Input:** Original raw pair CSVs (`pairs_raw_*.csv`).
* **Process:** Creates new CSV files containing pairs pointing to `.npy` files instead of `.wav` files.
* **Output:**
    * `pairs_raw_*_preprocessed_local.csv`: Contains paths to *local* `.npy` files (saved to Drive).
    * `pairs_raw_*_preprocessed_drive.csv`: Contains paths to *Drive* `.npy` files (saved to Drive).

## 4. Data Management (`data_manager_v3.py`)

This script handles copying data between Google Drive and the local Colab runtime using specific flags and dataset keys (`train_clean`, `train_noisy`, `test_clean`, etc.). Key actions relevant to the workflow:

* `--copy_raw`: Copies original WAV files from Drive to local.
* `--copy_spec`: Copies spectrogram PNG files from Drive to local.
* `--preprocess_raw`: Generates NPY from local WAVs (saves local+Drive).
* `--load_npy`: Copies NPY files from Drive to local.
* `--gen_csv_local`/`--gen_csv_drive`: Generates NPY pair CSVs.

## 5. Model Architectures

### 5.1. SincNet (`models/sincnet.py`)

* **Input:** Raw audio waveform (`.npy` format, preprocessed to fixed length).
* **Architecture:** Uses learnable sinc-based convolutional filters in the first layer, followed by standard Conv1D, MaxPool, LayerNorm/BatchNorm layers, and fully connected layers.
* **Output:** Fixed-size speaker embedding (e.g., 256 dimensions).

### 5.2. MobileNetV2 (`models/mobilenet_embedding.py`)

* **Input:** Mel-spectrogram image (PNG format, e.g., 224x224).
* **Architecture:** Uses a pre-trained MobileNetV2 (from `torchvision`) as a feature extractor. Features can be frozen or fine-tuned. An Adaptive Average Pooling layer and a final linear layer project features to the desired embedding size.
* **Output:** Fixed-size speaker embedding (e.g., 256 dimensions).

### 5.3. Fusion Classifier (`scripts/train_fusion.py`)

* **Input:** Concatenated embeddings from SincNet and MobileNetV2 for a pair of utterances `[e1_raw, e1_spec, e2_raw, e2_spec]`.
* **Architecture:** Multi-Layer Perceptron (MLP) with BatchNorm and ReLU/LeakyReLU activations, outputting a single logit (raw score before sigmoid).
* **Output:** Verification score (logit).

## 6. Model Training & Fine-tuning

Training follows a multi-stage process, orchestrated by the main workflow notebook:

1.  **Initial SincNet Training (`scripts/train_sincnet.py`):**
    * Input: Pre-processed *noisy* raw audio pairs (`pairs_raw_train_preprocessed_local.csv`).
    * Task: Train SincNet embedder + temporary PairClassifier head on genuine/imposter task.
    * Loss: `BCEWithLogitsLoss`.
    * Output: Best SincNet embedder weights (`sincnet_best.pt`).
2.  **Initial MobileNetV2 Training (`scripts/train_mobilenetv2.py`):**
    * Input: Regenerated high-quality *noisy* spectrogram pairs (`pairs_spec_train_local.csv`).
    * Task: Train MobileNetV2 embedder (typically freezing backbone features) + temporary PairClassifier head.
    * Loss: `BCEWithLogitsLoss`.
    * Output: Best MobileNetV2 embedder weights (`mobilenetv2_best.pt`).
3.  **Fusion Model Fine-tuning (`scripts/train_fusion.py`):**
    * Input: **Mixed** dataset combining *clean* and *noisy* pairs (`pairs_raw_mixed_train_preprocessed_local.csv` AND `pairs_spec_mixed_train_local.csv`).
    * Initialization: Loads best weights from initial SincNet and MobileNetV2 training.
    * Task: Fine-tune the **entire** network (SincNet + MobileNetV2 + Fusion Classifier) end-to-end. Embedders are **unfrozen** (`--no-freeze_embedders`).
    * Learning Rate: Uses a **lower** learning rate (e.g., 1e-5) than initial training.
    * Loss: `BCEWithLogitsLoss`.
    * Output: Best fine-tuned Fusion model checkpoint (`fusion_best.pt` saved in a separate directory like `outputs_finetuned`).

**Optimizations Used:**
* **Automatic Mixed Precision (AMP):** Enabled via `torch.amp.autocast` and `torch.amp.GradScaler`.
* **`torch.compile`:** Used (if PyTorch >= 2.0).
* **Data Preprocessing:** Raw audio pre-processed to `.npy`.
* **Local Data:** Training data copied locally.
* **DataLoader Tuning:** `num_workers`, `batch_size`, `persistent_workers=True`.

## 7. Evaluation (`scripts/evaluate.py` / Notebook Cell)

* **Models Evaluated:**
    1.  Original SincNet standalone (from initial noisy training).
    2.  Original MobileNetV2 standalone (from initial noisy training).
    3.  **Fine-tuned** Fusion Model (from mixed data fine-tuning).
* **Test Conditions:** Evaluation is performed separately on the three test sets using local data:
    1.  Clean (`pairs_raw_clean_test_preprocessed_local.csv`, `pairs_spec_clean_test_local.csv`)
    2.  Noisy (`pairs_raw_noisy_test_preprocessed_local.csv`, `pairs_spec_noisy_test_local.csv`)
    3.  Filtered (`pairs_raw_filtered_test_preprocessed_local.csv`, `pairs_spec_filtered_test_local.csv`)
* **Metrics:** The evaluation code calculates and saves:
    * Equal Error Rate (EER) and corresponding threshold.
    * Area Under the ROC Curve (AUC).
    * False Match Rate (FMR) at specific False Non-Match Rate (FNMR) targets (e.g., FNMR=1%, FNMR=0.1%).
* **Output:** Results are saved to a JSON file (e.g., `outputs_finetuned/metrics/evaluation_results_finetuned.json`).

## 8. Dependencies

* Python 3.x
* PyTorch (>= 1.7 recommended for `torch.amp`, >= 2.0 for `torch.compile`)
* torchvision
* torchaudio
* librosa
* soundfile
* numpy
* pandas
* scipy
* scikit-learn
* matplotlib (for saving spectrograms/analysis)
* Pillow (PIL)
* tqdm
* deepfilternet (for noise filtering stage)
* Google Colab environment (recommended)

