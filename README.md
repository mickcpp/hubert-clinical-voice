# HuBERT-based Clinical Voice Analysis for Neurodegenerative and Cognitive Disorders

> Deep learning workflow for binary classification of clinical voice recordings using HuBERT, including preprocessing, model training, validation, and preliminary explainability analysis.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mickcpp/hubert-clinical-voice/blob/main/notebooks/hubert_clinical_voice.ipynb)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Dataset Requirements](#dataset-requirements)
- [Configuration](#configuration)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Explainability (XAI)](#explainability-xai)
- [Adapting to Your Data](#adapting-to-your-data)
- [Results & Outputs](#results--outputs)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This repository contains the complete implementation of a **clinical voice analysis system** developed for a Bachelor's thesis, designed for the detection of neurodegenerative and cognitive disorders using speech biomarkers. The system leverages:

- **HuBERT** (Hidden-Unit BERT) transformer for self-supervised speech representation learning
- **Attention-based pooling** for sequence aggregation
- **5-fold stratified cross-validation** with composite early stopping
- **Integrated Gradients** and **Attention Rollout** for model interpretability

The code was designed for **Google Colab** execution with GPU acceleration (NVIDIA L4/T4/A100).

### Key Contributions

1. **Fine-tuning strategy**: Selective layer freezing (freeze layers 0-8, train 9-11)
2. **Data augmentation**: Conservative audio augmentation (Gaussian noise + gain)
3. **Onset trimming**: Energy-based speech onset detection to remove initial silence
4. **Composite early stopping**: Multi-metric optimization (AUC + Balanced Accuracy + Loss)
5. **Explainability**: Integrated Gradients for attributions and Attention Rollout for focus

---

## ✨ Features

### Core Capabilities

- ✅ **End-to-end training pipeline** from raw audio to trained classifier
- ✅ **Automatic Mixed Precision (AMP)** for faster training and lower memory usage
- ✅ **Class-weighted loss** to handle imbalanced datasets
- ✅ **Learning rate scheduling** with ReduceLROnPlateau
- ✅ **Comprehensive metrics**: Accuracy, Balanced Accuracy, F1-Score, AUC-ROC
- ✅ **Detailed visualizations**: Learning curves, ROC curves, confusion matrices
- ✅ **Dataset analysis tools**: Audio quality assessment, onset detection, duration statistics
- ✅ **Model interpretability**: Integrated Gradients with temporal attribution maps / Attention Rollout for focus

### Modular Execution

The notebook uses **execution flags** to control which sections run:

```python
RUN_DATASET_ANALYSIS = False  # Audio quality analysis
RUN_TRAINING_CV = True        # 5-fold cross-validation training
RUN_XAI = False               # Explainability analysis
```

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. Click the "Open in Colab" badge above
2. Mount your Google Drive (when prompted)
3. Upload your dataset to Drive following the [structure below](#dataset-requirements)
4. Update the base path in **Cell 3**:
   ```python
   BASE_PATH = '/content/drive/MyDrive/YOUR_FOLDER/'
   ```
5. Run all cells sequentially

### Option 2: Local Execution

⚠️ **Note**: While this code can run locally, it was optimized for Colab. Local execution requires:
- NVIDIA GPU with CUDA support (16GB+ VRAM recommended)
- Manual creation of directory structure
- Adaptation of Google Drive paths

---

## 📦 Installation

### Dependencies

The notebook automatically installs required packages in **Cell "Setup experiment"**:

```bash
pip install transformers==4.47.1
pip install librosa==0.10.2
pip install audiomentations==0.35.0
```

### Full Requirements

For local setup, install all dependencies:

```bash
pip install -r requirements.txt
```

**Core libraries**:
- `torch>=2.0.0` - Deep learning framework
- `transformers==4.47.1` - HuBERT model from Hugging Face
- `librosa==0.10.2` - Audio processing
- `audiomentations==0.35.0` - Data augmentation
- `scikit-learn>=1.0` - Cross-validation and metrics
- `pandas>=1.3.0` - Data manipulation
- `numpy>=1.21.0` - Numerical computing
- `matplotlib>=3.5.0` - Plotting
- `tqdm>=4.62.0` - Progress bars

---

## 📊 Dataset Requirements

### ⚠️ Privacy Notice

**The dataset used in this thesis cannot be shared due to medical privacy constraints.** To use this code, you must provide your own audio data.

### Required Directory Structure

```
YOUR_BASE_FOLDER/
├── data/
│   ├── raw/
│   │   ├── dataset_free_speech.csv        # Metadata CSV
│   │   └── audio/                         # Audio files
│   │       ├── file1.wav
│   │       ├── file2.wav
│   │       └── ...
│   └── processed/                         # (Created automatically)
├── models/
│   └── YOUR_EXPERIMENT_NAME/              # (Created automatically)
│       ├── best_model_fold1.pth
│       ├── best_model_fold2.pth
│       └── ...
└── results/
    └── YOUR_EXPERIMENT_NAME/              # (Created automatically)
        ├── cv_results.csv
        ├── learning_curves.png
        └── ...
```

### CSV Format

**Required columns**:

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| `FileName` | string | Audio filename | `patient_001.wav` |
| `Tipo soggetto` | string | Subject type | `Controllo` or `Paziente` |

**CSV file settings**:
- **Separator**: `;` (semicolon)
- **Encoding**: UTF-8
- **Header**: Yes (first row)

**Example CSV**:
```csv
FileName;Tipo soggetto;Other_Column
control_01.wav;Controllo;metadata
patient_01.wav;Paziente;metadata
control_02.wav;Controllo;metadata
```

### Audio File Requirements

- **Format**: WAV (mono or stereo, will be converted to mono)
- **Sample rate**: Any (will be resampled to 16kHz for HuBERT)
- **Duration**: Variable (recommended 20-60 seconds)
  - Files longer than 30s will be truncated
  - Very short files (<5s after onset trimming) will trigger warnings
- **Bit depth**: 16-bit or 32-bit PCM recommended
- **Quality**: Clean recordings, minimal background noise

### Labels

The code creates binary labels:
- **0**: Control (`Controllo`)
- **1**: Patient (`Paziente`)

If your data uses different label names, modify **Cell "CARICAMENTO DATASET"** (line 428):
```python
# Original:
df['label'] = (df['Tipo soggetto'] == 'Paziente').astype(int)

# Adapt to your labels (e.g., 'Healthy' vs 'Diseased'):
df['label'] = (df['Tipo soggetto'] == 'Diseased').astype(int)
```

---

## ⚙️ Configuration

### Experiment Setup (Cell "Setup experiment")

```python
EXPERIMENT_NAME = 'hubert_attention_pooling'  # Name for your experiment
```

All models and results will be saved under this experiment name.

### Main Configuration Dictionary

The `CONFIG` dictionary (Cell after "Setup experiment") controls all hyperparameters:

```python
CONFIG = {
    # ===== DATA =====
    'csv_path': DATA_RAW_PATH + 'dataset_free_speech.csv',
    'audio_dir': DATA_RAW_PATH + 'audio/',
    
    # ===== PREPROCESSING =====
    'target_sr': 16000,        # HuBERT requires 16kHz
    'max_duration': 30,        # Truncate audio longer than 30s
    
    # ===== MODEL =====
    'model_name': 'facebook/hubert-base-ls960',  # Pretrained HuBERT checkpoint
    'freeze_layers': 9,        # Freeze layers 0-8, fine-tune 9-11
    'pooling_type': 'attention',  # 'attention' or 'mean'
    
    # ===== TRAINING =====
    'n_folds': 5,              # K-fold cross-validation
    'batch_size': 4,           # Adjust based on GPU memory
    'num_epochs': 30,          # Maximum training epochs
    'learning_rate': 5e-5,     # Standard fine-tuning LR
    'weight_decay': 5e-4,      # L2 regularization
    
    # ===== LR SCHEDULER =====
    'scheduler': {
        'factor': 0.5,         # Reduce LR by 50% when plateau
        'patience': 3,         # Wait 3 epochs before reducing
        'min_lr': 1e-6,        # Minimum learning rate
    },
    
    # ===== CLASSIFIER HEAD =====
    'hidden_dim': 256,         # MLP hidden dimension
    'dropout': 0.25,           # Dropout probability
    'num_classes': 2,          # Binary classification
    
    # ===== EARLY STOPPING =====
    'early_stopping': {
        'warmup_epochs': 8,           # Initial warmup period
        'max_loss_threshold': 0.75,   # skip if loss explodes
        'use_composite_score': True,  # Multi-metric optimization
        'composite_weights': {
            'auc': 0.45,              # 45% weight to AUC-ROC
            'balacc': 0.35,           # 35% weight to Balanced Accuracy
            'loss': -0.2,             # -20% weight (penalize high loss)
        },
        'patience': 10,               # Stop after 10 epochs without improvement
    },
    
    # ===== AUGMENTATION =====
    'use_augmentation': True,  # Enable audio augmentation
    'aug_prob': 0.165,         # Probability of applying each augmentation
    
    # ===== REPRODUCIBILITY =====
    'seed': 42,                # Random seed for reproducibility
}
```

### Tuning Guidelines

**Memory-constrained GPU** (8-12GB VRAM):
```python
'batch_size': 2,           # Reduce batch size
'freeze_layers': 10,       # Freeze more layers (train only layer 11)
```

**Large GPU** (24GB+ VRAM):
```python
'batch_size': 8,           # Increase batch size
'freeze_layers': 6,        # Fine-tune more layers (7-11)
```

**Different label ratio** (e.g., 1:3 imbalance):
- Class weights are computed automatically in the training loop
- No manual adjustment needed

**Different audio lengths**:
```python
'max_duration': 60,
```

---

## 🎓 Usage

### 1. Dataset Analysis (Optional)

Enable detailed audio quality analysis:

```python
RUN_DATASET_ANALYSIS = True
RUN_TRAINING_CV = False
RUN_XAI = False
```

**What it does**:
- Computes duration statistics (total and post-trimming)
- Detects speech onset using energy-based method
- Measures silence ratio, RMS energy, zero-crossing rate
- Identifies clipping and problematic files
- Generates comprehensive visualizations

**Main Outputs**:
- `data/dataset_free_speech_analysis/audio_statistics.csv` - Per-file metrics
- `data/dataset_free_speech_analysis/audio_analysis.png` - Statistical plots
- `data/dataset_free_speech_analysis/problematic_audio.csv` - Critical audio

**When to use**:
- First time working with a new dataset
- To identify problematic recordings
- To generate onset trimming map (optional feature)

### 2. Training with Cross-Validation

Enable 5-fold stratified cross-validation:

```python
RUN_DATASET_ANALYSIS = False
RUN_TRAINING_CV = True
RUN_XAI = False
```

**What it does**:
- Splits data into 5 stratified folds
- Trains a separate model for each fold
- Validates on held-out fold
- Computes per-fold and aggregate metrics
- Saves best model for each fold

**Outputs** (saved to `results/YOUR_EXPERIMENT_NAME/`):
- `cv_results.csv` - Per-fold metrics
- `learning_curves.png` - Training/validation curves for all folds
- `roc_cm_final.png` - Mean ROC curve + aggregated confusion matrix
- Model checkpoints: `models/YOUR_EXPERIMENT_NAME/best_model_fold{1-5}.pth`

**Training time**:
- ~45-60 minutes per fold on NVIDIA L4 GPU
- ~2-3 hours total for 5-fold CV

### 3. Explainability Analysis

Analyze model predictions with Integrated Gradients and Attention Rollout:

```python
RUN_DATASET_ANALYSIS = False
RUN_TRAINING_CV = False
RUN_XAI = True
```

**Prerequisites**:
1. Training must be completed (models must exist)
2. You must select which fold's model to analyze

**What it does**:
- Loads best model from a specific fold
- Retrieves corresponding validation set
- Computes Integrated Gradients and Attention Rollout for selected samples
- Generates temporal attribution maps and temporal attention maps overlaid on spectrograms/waveforms

**Outputs** (saved to `results/YOUR_EXPERIMENT_NAME/explainability/`):
- `integrated_gradients/IG_{sample_name}.png` - Spectrogram + attribution map for each analyzed sample
- `attention_rollout/Rollout_{sample_name}.png` - Spectrogram/Waveforms + attention maps for each analyzed sample

- Integrated Gradients: Provides temporal attributions, showing which parts of the input truly contributed to the model's prediction
- Attention Rollout: Reveals the temporal regions the model attended to during processing, indicating where the model “looked”

**Customization**:
Select different fold for XAI (see [Adapting to Your Data](#adapting-to-your-data) section).

---

## 🏗️ Model Architecture

### HuBERTClassifier

```
Input Audio (16kHz WAV)
    ↓
[Wav2Vec2FeatureExtractor]
    ↓
HuBERT Encoder (12 transformer layers, 768-d)
    ├─ Layers 0-8: FROZEN (general acoustic features)
    └─ Layers 9-11: FINE-TUNED (task-specific adaptation)
    ↓
Sequence Embeddings: (batch, time, 768)
    ↓
[Attention Pooling] or [Mean Pooling]
    ↓
Aggregated Embedding: (batch, 768)
    ↓
MLP Classifier:
    ├─ Dropout(0.25)
    ├─ Linear(768 → 256)
    ├─ ReLU
    ├─ Dropout(0.25)
    └─ Linear(256 → 2)
    ↓
Output Logits: (batch, 2)
```

### Attention Pooling

Instead of averaging all timesteps equally (mean pooling), **attention pooling** learns to weight important frames more heavily:

```python
attention_weights = softmax(Linear(embeddings))  # (batch, time, 1)
pooled = sum(attention_weights * embeddings)      # (batch, 768)
```

**Benefits**:
- Focuses on discriminative speech segments
- Ignores silence/noise
- Improves performance on variable-length audio

### Preprocessing Pipeline

Each audio file undergoes:

1. **Loading**: Librosa loads at 16kHz mono
2. **Onset Trimming** (optional): Remove initial silence based on energy detection
3. **Augmentation** (training only):
   - Gaussian noise (0.2-1.5% amplitude)
   - Volume gain (±6 dB)
4. **Truncation**: Crop to 30 seconds max
5. **Feature Extraction**: Wav2Vec2FeatureExtractor normalizes waveform

---

## 🚂 Training Pipeline

### 5-Fold Stratified Cross-Validation

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold in [1, 2, 3, 4, 5]:
    train_data, val_data = split(fold)
    model = HuBERTClassifier(...)
    
    for epoch in range(30):
        train_loss = train_epoch(...)
        val_metrics = validate_epoch(...)
        
        if early_stopping_triggered:
            break
    
    save_best_model(f"best_model_fold{fold}.pth")
```

**Why stratified**:
- Maintains same control/patient ratio in each fold
- Critical for imbalanced datasets

### Two-Phase Early Stopping

#### Phase 1: Warmup (Epochs 1-8)
- **Goal**: Find a good baseline model
- **Criterion**: Save model if `loss < 0.75` AND `AUC improves`
- **Rationale**: Prevent saving models before loss stabilizes

#### Phase 2: Composite Score (Epochs 9+)
- **Goal**: Optimize multiple metrics simultaneously
- **Formula**:
  ```python
  score = 0.45 * AUC + 0.35 * BalAcc - 0.2 * Loss
  ```
- **Criterion**: Save model if `score improves` AND `loss < 0.75`
- **Stop**: If no improvement for 10 consecutive epochs

**Why composite score**:
- AUC alone can be misleading with imbalanced data
- Balanced Accuracy ensures performance on both classes
- Loss penalty prevents overfitting

### Class-Weighted Loss

Automatically handles class imbalance:

```python
weight_control = n_samples / (2 * n_controls)
weight_patient = n_samples / (2 * n_patients)

criterion = CrossEntropyLoss(weight=[weight_control, weight_patient])
```

**Example**: Dataset with 54 controls, 79 patients:
- Control weight: 1.23
- Patient weight: 0.84
- Effect: Penalize misclassifying minority class more

### Automatic Mixed Precision (AMP)

Reduces memory usage and speeds up training:

```python
scaler = GradScaler()

with autocast():
    logits = model(inputs)  # Forward in fp16
    loss = criterion(logits, labels)

scaler.scale(loss).backward()  # Scaled gradients
scaler.step(optimizer)
scaler.update()
```

**Benefits**:
- ~40% faster training
- ~30% lower memory usage
- No accuracy loss

---

## 🔍 Explainability (XAI)

### Integrated Gradients

**Method**: Computes attribution by integrating gradients along the path from a baseline (silence) to the actual input.

**Formula**:
```
Attribution(x) = (x - baseline) × ∫₀¹ ∇F(baseline + α(x - baseline)) dα
```

**Implementation highlights**:

1. **Non-uniform alpha sampling**: Uses power scaling (α³) to concentrate samples near baseline
   - More accurate for models with high curvature near zero input

2. **Adaptive integration steps**:
   - High confidence (>95%): 600 steps
   - Medium confidence (85-95%): 500 steps
   - Normal confidence (<85%): 30 steps

3. **Completeness check**: Verifies attribution sums match prediction difference
   ```python
   error = |sum(attributions) - (F(input) - F(baseline))|
   ```

### Visualization

Generates two-panel plot:
1. **Top**: Mel-spectrogram (standard acoustic representation)
2. **Bottom**: Attribution map (red = positive attribution, blue = negative)

**Interpretation**:
- **Red regions**: Time segments pushing prediction toward target class
- **Blue regions**: Time segments opposing target class prediction
- **Temporal patterns**: Can reveal speech characteristics (pauses, specific phonemes, prosody)

### Use Cases

- **Clinical insights**: Identify discriminative speech patterns
- **Sample selection**: Find prototypical examples for each class

---

## 🔧 Adapting to Your Data

### Critical Changes (MUST MODIFY)

#### 1. Base Path (Cell 3: "Configure Paths")

**Line ~227**:
```python
# CHANGE THIS to your Google Drive folder
BASE_PATH = '/content/drive/MyDrive/Tesi/'
```

**To**:
```python
BASE_PATH = '/content/drive/MyDrive/YOUR_PROJECT_FOLDER/'
```

#### 2. CSV Metadata (Cell: "CARICAMENTO DATASET")

**Line ~428** - If your CSV uses different label names:
```python
# Original:
df['label'] = (df['Tipo soggetto'] == 'Paziente').astype(int)

# Adapt to your labels:
df['label'] = (df['YOUR_COLUMN'] == 'YOUR_POSITIVE_CLASS').astype(int)
```

**Line ~425** - If using different CSV separator:
```python
# Original:
df = pd.read_csv(CONFIG['csv_path'], sep=';')

# For comma-separated:
df = pd.read_csv(CONFIG['csv_path'], sep=',')
```

#### 3. XAI Fold Selection (Cell: "Recupero validation set")

**Line ~3373** - Change target fold:
```python
# Original (analyzes fold 4):
if fold_num == 4:
    val_df_fold4 = df.iloc[val_idx].reset_index(drop=True)
    break

# To analyze fold 2:
if fold_num == 2:
    val_df_fold2 = df.iloc[val_idx].reset_index(drop=True)
    break
```

**Line ~3225 and ~3355** - Update model path:
```python
# Original:
MODEL_PATH = os.path.join(paths['models'], 'final_model', 'best_model_fold4.pth')

# Change to your fold:
MODEL_PATH = os.path.join(paths['models'], 'final_model', 'best_model_fold2.pth')
```

### Important Changes (SHOULD MODIFY)

#### 4. Problematic Files Exclusion (Cell: "CARICAMENTO DATASET")

**Line ~433** - Update with files to exclude from your dataset:
```python
# Original (thesis-specific):
problematic_files = [
    'D_AP_F_51_2024_10_23_Italian.wav'
]

# Your problematic files (or empty list):
problematic_files = [
    'noisy_recording_01.wav',
    'truncated_file_05.wav'
]

# Or disable entirely:
problematic_files = []
```

#### 5. Onset Trimming Map (Cell: "CARICAMENTO DATASET")

**Line ~447** - Optional feature for removing initial silence:

**Option A**: Use automated onset detection (recommended)
1. Run dataset analysis first (`RUN_DATASET_ANALYSIS = True`)
2. This generates `audio_statistics.csv` with onset times
3. Code automatically loads this file if it exists

**Option B**: Disable onset trimming
```python
# Comment out or modify:
onset_stats_path = os.path.join(DATA_PATH, 'dataset_free_speech_analysis', 'audio_statistics.csv')

if os.path.exists(onset_stats_path):
    # ... onset loading code ...
else:
    print(f"\n⚠ WARNING: Onset map non trovato")
    onset_map = {}  # Empty map = no trimming

# To force disable:
onset_map = {}  # Add this line to override
```

### Optional Tuning

#### 6. Hyperparameters (CONFIG dictionary)

Adjust based on your dataset size and class balance:

```python
# Small dataset (<100 samples):
CONFIG['num_epochs'] = 20          # Reduce risk of overfitting
CONFIG['dropout'] = 0.35           # Increase dropout
CONFIG['freeze_layers'] = 10       # Freeze more layers

# Large dataset (>500 samples):
CONFIG['num_epochs'] = 50          # Allow longer training
CONFIG['batch_size'] = 8           # Larger batches
CONFIG['freeze_layers'] = 6        # Fine-tune more layers

# Extreme imbalance (1:5 ratio):
# Class weights handle this automatically, but you can:
CONFIG['early_stopping']['composite_weights']['balacc'] = 0.50  # Prioritize balance
CONFIG['early_stopping']['composite_weights']['auc'] = 0.30
```

#### 7. Augmentation Settings

**Disable augmentation** (conservative approach):
```python
CONFIG['use_augmentation'] = False
```

**Increase augmentation** (larger dataset, more regularization needed):
```python
CONFIG['aug_prob'] = 0.3  # 30% chance per augmentation
```

**Customize augmentation types** (Cell: "SpeechDataset"):
```python
# Line ~512 - Modify augmentor composition:
from audiomentations import Compose, AddGaussianNoise, Gain, TimeStretch, PitchShift

self.augmentor = Compose([
    AddGaussianNoise(min_amplitude=0.002, max_amplitude=0.015, p=aug_prob),
    Gain(min_gain_in_db=-6, max_gain_in_db=6, p=aug_prob*0.85),
    TimeStretch(min_rate=0.95, max_rate=1.05, p=aug_prob*0.5),  # NEW
    PitchShift(min_semitones=-2, max_semitones=2, p=aug_prob*0.3),  # NEW
])
```

---

## 📈 Results & Outputs

### Cross-Validation Results

**File**: `results/YOUR_EXPERIMENT_NAME/cv_results.csv`

| Fold | Val Accuracy | Val Balanced Acc | Val F1 | Val AUC | TN | FP | FN | TP |
|------|--------------|------------------|--------|---------|----|----|----|----|
| 1 | 0.8519 | 0.8462 | 0.8654 | 0.9091 | 9 | 2 | 2 | 14 |
| 2 | 0.8889 | 0.8846 | 0.8958 | 0.9359 | 10 | 1 | 2 | 14 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

### Visualizations

#### 1. Learning Curves (`learning_curves.png`)

Four-panel plot showing:
- **Top-left**: Training loss (all folds)
- **Top-right**: Validation loss (all folds)
- **Bottom-left**: Validation balanced accuracy (all folds)
- **Bottom-right**: Validation AUC-ROC (all folds)

**Use to**:
- Detect overfitting (train/val gap)
- Identify best-performing folds
- Verify learning stability

#### 2. ROC Curve + Confusion Matrix (`roc_cm_final.png`)

Two-panel plot:
- **Left**: Mean ROC curve across all folds with ±1 std band
- **Right**: Aggregated confusion matrix (sum of all fold predictions)

**Use to**:
- Report final model performance
- Visualize trade-off between sensitivity/specificity
- Identify class-specific errors

#### 3. Explainability Plots (`explainability/IG_*.png`)

Per-sample visualization:
- **Top**: Mel-spectrogram
- **Bottom**: Integrated Gradients attribution map

**Use to**:
- Understand model decisions
- Validate clinical relevance

### Model Checkpoints

**Location**: `models/YOUR_EXPERIMENT_NAME/best_model_fold{1-5}.pth`

**Contents**:
```python
checkpoint = {
    'model_state_dict': ...,     # Model weights
    'scaler_state_dict': ...,    # AMP scaler state
    'epoch': 15,                 # Best epoch
    'val_loss': 0.4234,
    'val_auc': 0.8956,
    'val_bal_acc': 0.8723,
    'composite_score': 0.7234,   # (if using composite early stopping)
}
```

**Load model**:
```python
model = HuBERTClassifier(...)
checkpoint = torch.load('best_model_fold1.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 📁 Project Structure

```
YOUR_PROJECT_FOLDER/
├── hubert_clinical_voice.ipynb        # Main notebook
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── data/
│   ├── raw/
│   │   ├── dataset_free_speech.csv
│   │   └── audio/*.wav
│   ├── processed/                     # (auto-created)
│   └── dataset_free_speech_analysis/  # (if RUN_DATASET_ANALYSIS=True)
│       ├── audio_statistics.csv
│       └── audio_analysis.png
│
├── models/
│   └── YOUR_EXPERIMENT_NAME/
│       ├── best_model_fold1.pth
│       ├── best_model_fold2.pth
│       ├── ...
│       └── final_model/               # (for XAI)
│           └── best_model_fold4.pth
│
└── results/
    └── YOUR_EXPERIMENT_NAME/
        ├── cv_results.csv
        ├── learning_curves.png
        ├── roc_cm_final.png
        └── explainability/            # (if RUN_XAI=True)
            ├── ...

```

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. "Drive not mounted" error
```
Solution: Run the first cell "Mount Google Drive" and authorize access
```

#### 2. "File not found" for audio files
```
Checklist:
- Verify FileName column in CSV matches actual filenames
- Check audio_dir path points to correct folder
- Ensure files have .wav extension
- Files are in /content/drive/MyDrive/YOUR_FOLDER/data/raw/audio/
```

#### 3. Out of Memory (OOM) error
```
Solutions:
1. Reduce batch size: CONFIG['batch_size'] = 2
2. Freeze more layers: CONFIG['freeze_layers'] = 10
3. Restart runtime and clear GPU cache: Runtime > Restart Runtime
```

#### 4. Training very slow
```
Checklist:
- GPU enabled? Runtime > Change runtime type > GPU (T4/L4/A100)
- AMP enabled? (should be by default)
```

#### 5. "Onset map not found" warning
```
This is OPTIONAL - you can:
- Ignore (onset trimming disabled, code still works)
- Run dataset analysis first to generate it
- Comment out onset loading code (see Section 5 in Adapting)
```

#### 6. XAI cell fails "Model not found"
```
Checklist:
- Did you run training first? (RUN_TRAINING_CV = True)
- Check model path matches fold number
- Models should be in: models/YOUR_EXPERIMENT_NAME/best_model_fold{N}.pth
- Verify EXPERIMENT_NAME is consistent
```

#### 7. RuntimeError: "CUDA out of memory" during XAI
```
Solutions:
1. Reduce n_steps in Integrated Gradients:
   - High confidence: 300 instead of 600
   - Normal: 20 instead of 30
2. Reduce internal_batch_size to 1
3. Analyze fewer samples
```

#### 8. Plots not displaying in Colab
```
Solutions:
- Ensure %matplotlib inline is executed (Cell 5)
- Try: plt.show() after each plot
- Check if plt.savefig() succeeded (files in results/)
```

