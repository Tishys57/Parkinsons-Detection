# EEG + fMRI Multimodal Classification with Deep Feature Fusion

A multimodal deep learning and machine learning pipeline for binary classification using **EEG and fMRI-derived features**. The project combines modality-specific neural networks, deep feature extraction, optimized feature fusion, Random Forest classification, cross-validation, and SHAP-based explainability.

The project explores how complementary information from EEG and fMRI can be combined into a single predictive representation while retaining interpretability through post-hoc feature analysis.

---

## Overview

This project implements an end-to-end multimodal classification pipeline:

```text
EEG Data ──► EEGNet-inspired CNN ──► Deep EEG Features ──┐
                                                          │
                                                          ├──► Weighted Feature Fusion
                                                          │         │
fMRI Data ──► CNN ─────────────────► Deep fMRI Features ─┘         │
                                                                    ▼
                                                          Random Forest Classifier
                                                                    │
                                                                    ▼
                                                           Classification Results
                                                                    │
                                                                    ▼
                                                              SHAP Analysis
                                                                    │
                                                                    ▼
                                                         Feature Selection / Retraining
```

Rather than relying directly on handcrafted EEG or fMRI features, the neural networks learn intermediate representations that are subsequently fused and passed to a classical machine learning classifier.

---

## Key Results

The final multimodal pipeline produced the following cross-validation results:

| Metric                          |              Result |
| ------------------------------- | ------------------: |
| Mean 10-fold CV Accuracy        | **0.7542 ± 0.0732** |
| Best fusion weight α            |            **0.76** |
| Best fusion CV accuracy         |          **0.7313** |
| Best Random Forest CV F1        |          **0.7816** |
| Overall classification accuracy |            **0.75** |
| Class 0 F1                      |            **0.75** |
| Class 1 F1                      |            **0.76** |

The optimized fusion weight was:

```text
α = 0.76
```

where the fused representation is constructed as:

```text
Fused Features = [0.76 × EEG Features | 0.24 × fMRI Features]
```

This indicates that, within the implemented optimization procedure, the selected fusion configuration assigned a larger weight to the EEG representation than to the fMRI representation.

### Classification Report

The final classification report contained balanced support across the two classes:

| Class             | Precision |   Recall |       F1 | Support |
| ----------------- | --------: | -------: | -------: | ------: |
| 0                 |      0.77 |     0.72 |     0.75 |     120 |
| 1                 |      0.74 |     0.78 |     0.76 |     120 |
| **Macro Avg.**    |  **0.76** | **0.75** | **0.75** | **240** |
| **Weighted Avg.** |  **0.76** | **0.75** | **0.75** | **240** |

The individual cross-validation fold accuracies varied, with the reported tenth fold achieving:

```text
Fold 10 accuracy: 0.8750
```

The mean across all ten folds was:

```text
Mean accuracy: 0.7542 ± 0.0732
```

The standard deviation is included to show the variation in performance across folds rather than reporting only the aggregate accuracy.

---

## SHAP-Based Feature Analysis

To investigate which learned features contributed most strongly to the Random Forest predictions, SHAP-based feature importance analysis was performed.

The top-ranked features were predominantly extracted from the EEG representation:

| Rank | Feature      | Importance |
| ---: | ------------ | ---------: |
|    1 | EEG_feat_143 |   0.017346 |
|    2 | EEG_feat_132 |   0.016790 |
|    3 | EEG_feat_135 |   0.012280 |
|    4 | EEG_feat_170 |   0.008724 |
|    5 | EEG_feat_156 |   0.007140 |
|    6 | EEG_feat_159 |   0.006453 |
|    7 | EEG_feat_131 |   0.006027 |
|    8 | EEG_feat_161 |   0.005944 |
|    9 | EEG_feat_183 |   0.005714 |
|   10 | EEG_feat_104 |   0.005351 |

Additional highly ranked features included:

```text
EEG_feat_302
EEG_feat_317
EEG_feat_305
EEG_feat_179
EEG_feat_115
EEG_feat_94
EEG_feat_136
EEG_feat_318
EEG_feat_191
EEG_feat_247
```

These feature names correspond to dimensions of the learned neural-network representation rather than directly interpretable physiological measurements. Therefore, their importance should be interpreted as importance of learned latent features, not as direct evidence that a particular EEG channel or brain region is biologically responsible for the classification.

---

## SHAP-Guided Feature Reduction

The SHAP analysis was subsequently used to select the most informative learned features and retrain the Random Forest classifier.

The retrained model used the selected SHAP features and obtained:

```text
Best RF parameters:
{
    'max_depth': 10,
    'min_samples_leaf': 1,
    'min_samples_split': 5,
    'n_estimators': 300
}

Best CV F1 score: 0.8028
```

For comparison, the original Random Forest optimization produced:

```text
Best RF parameters:
{
    'max_depth': 20,
    'min_samples_leaf': 1,
    'min_samples_split': 5,
    'n_estimators': 300
}

Best CV F1 score: 0.7816
```

Thus, within the reported 5-fold cross-validation procedure, the SHAP-guided feature reduction increased the best CV F1 score from:

```text
0.7816 → 0.8028
```

This is an improvement of **0.0212 F1 points** in the reported cross-validation score.

Importantly, the `0.8028` value is a **cross-validation F1 score**, not an independently held-out test-set result. Further evaluation on an untouched test set would be required to establish whether the improvement generalizes beyond the cross-validation procedure.

---

# Methodology

## 1. Data Loading

The project expects EEG and fMRI feature files in a Kaggle-style environment:

```text
/kaggle/input/eeg-features/eeg_features_labels.npy
/kaggle/input/fmri-features/fmri_features.npy
```

The EEG file contains:

```python
{
    "X": EEG_features,
    "y": labels
}
```

The EEG features, fMRI features, and labels are aligned by truncating them to the minimum available number of samples.

The experiments use fixed random seeds:

```python
np.random.seed(42)
tf.random.set_seed(42)
```

---

# EEG Processing

The EEG pipeline assumes 64 EEG channels.

The input is converted into a consistent representation:

```text
(samples, channels, timepoints, 1)
```

A `RobustScaler` is used during preprocessing, followed by reshaping into a format suitable for convolutional processing.

The default configuration uses:

| Parameter        | Value |
| ---------------- | ----: |
| EEG Channels     |    64 |
| Temporal Samples |   891 |
| F1               |    16 |
| F2               |    32 |
| Dropout          |   0.4 |
| Learning Rate    | 0.001 |

---

## EEG Model

The EEG model is inspired by the **EEGNet architecture** and uses a sequence of temporal, depthwise, and separable convolutions.

The architecture includes:

```text
Input
  │
  ▼
Reshape
  │
  ▼
Temporal Conv2D
  │
  ▼
Batch Normalization
  │
  ▼
Depthwise Conv2D
  │
  ▼
Batch Normalization
  │
  ▼
ELU
  │
  ▼
Average Pooling
  │
  ▼
Dropout
  │
  ▼
Separable Conv2D blocks
  │
  ▼
Batch Normalization + ELU
  │
  ▼
Average Pooling + Dropout
  │
  ▼
Flatten
  │
  ▼
Dense(128)
  │
  ▼
"features"
  │
  ▼
Binary Prediction
```

The intermediate dense layer named `features` is used as the learned EEG representation for multimodal fusion.

---

# fMRI Processing

The fMRI feature vectors are converted into a 2D representation for CNN processing.

The preprocessing pipeline:

1. Attempts to reshape the feature vector into a square.
2. Truncates or pads when required.
3. Resizes the resulting representation to `64 × 64`.
4. Adds a single channel dimension.

The resulting representation has the form:

```text
(samples, 64, 64, 1)
```

This project therefore treats the provided fMRI feature vectors as computational 2D representations. It does **not** directly process the original 3D fMRI volumes.

---

## fMRI CNN

The fMRI network contains three convolutional blocks:

```text
Input
  │
  ▼
Conv2D(64)
  │
BatchNorm
  │
MaxPooling
  │
Dropout
  │
  ▼
Conv2D(128)
  │
BatchNorm
  │
MaxPooling
  │
Dropout
  │
  ▼
Conv2D(256)
  │
BatchNorm
  │
MaxPooling
  │
Dropout
  │
  ▼
Global Average Pooling
  │
  ▼
Dense(256) → "features"
  │
  ▼
Dropout
  │
  ▼
Binary Prediction
```

The intermediate `features` layer provides the learned fMRI representation used during multimodal fusion.

---

# Training Strategy

Both neural networks use regularization and training callbacks intended to reduce overfitting.

Techniques include:

* Dropout
* Batch normalization
* L2 kernel regularization
* Max-norm constraints
* Early stopping
* Learning-rate reduction
* Model checkpointing
* Class-weighted training

The class weights are calculated using:

```python
compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y),
    y=y
)
```

### Training Callbacks

**EarlyStopping**

```text
patience = 15
restore_best_weights = True
```

**ReduceLROnPlateau**

```text
patience = 7
factor = 0.5
```

Best models are saved using:

```text
best_eeg_model.h5
best_fmri_model.h5
```

---

# Multimodal Feature Fusion

After training the modality-specific networks, their intermediate representations are extracted.

Let:

* `E` = normalized EEG feature representation
* `F` = normalized fMRI feature representation
* `α` = EEG fusion weight

The fused representation is:

```text
Fused = [αE | (1 − α)F]
```

The extracted EEG and fMRI representations are independently normalized using `RobustScaler` before fusion.

---

# Fusion Weight Optimization

Instead of selecting the EEG/fMRI contribution manually, the project uses **Optuna** to optimize the fusion coefficient.

The search space is:

```text
α ∈ [0.01, 0.99]
step = 0.01
```

The optimization objective is stratified cross-validation accuracy using a Random Forest classifier.

The reported optimum was:

```text
Best alpha: 0.76
Best CV Score: 0.7313
```

Therefore:

```text
EEG contribution  = 76%
fMRI contribution = 24%
```

within the weighted feature construction used by this experiment.

---

# Random Forest Classification

The fused deep representations are classified using a Random Forest.

The model undergoes hyperparameter optimization using `GridSearchCV`.

### Search Space

```python
n_estimators = [100, 200, 300]

max_depth = [10, 20, None]

min_samples_split = [2, 5]

min_samples_leaf = [1, 2]
```

The optimization uses:

```text
5-fold cross-validation
scoring = weighted F1
```

The initial optimized configuration was:

```text
n_estimators      = 300
max_depth         = 20
min_samples_split = 5
min_samples_leaf  = 1

Best CV F1 = 0.7816
```

---

# Cross-Validation

The overall multimodal evaluation uses **10-fold StratifiedKFold cross-validation**.

Each fold follows the general pipeline:

```text
Split data
   │
   ├── EEG training data
   └── fMRI training data
           │
           ▼
Train modality-specific neural networks
           │
           ▼
Extract deep features
           │
           ▼
Normalize features
           │
           ▼
Optimize fusion weight
           │
           ▼
Fuse EEG + fMRI features
           │
           ▼
Optimize Random Forest
           │
           ▼
Predict held-out fold
           │
           ▼
Record metrics
```

The reported mean cross-validation accuracy was:

```text
0.7542 ± 0.0732
```

---

# Explainability with SHAP

The project uses **SHAP (SHapley Additive exPlanations)** to investigate the contribution of learned features to the Random Forest predictions.

The analysis generates feature-level importance values and ranks features according to their mean absolute SHAP contribution.

Feature names are constructed as:

```text
EEG_feat_i
fMRI_feat_i
```

This makes it possible to inspect which dimensions of the learned multimodal representation have the greatest influence on the classifier.

The top-ranked features in the reported analysis were primarily EEG-derived features.

---

# SHAP-Guided Retraining

The project goes beyond simply visualizing feature importance.

The highest-ranked SHAP features are selected and used to construct a reduced representation. The Random Forest is then retrained on this feature subset.

The resulting optimized model achieved:

```text
Best CV F1 = 0.8028
```

with:

```text
n_estimators      = 300
max_depth         = 10
min_samples_split = 5
min_samples_leaf  = 1
```

This creates an iterative pipeline:

```text
Deep Feature Extraction
        ↓
Multimodal Fusion
        ↓
Random Forest
        ↓
SHAP Explainability
        ↓
Feature Selection
        ↓
Reduced Feature Representation
        ↓
Random Forest Retraining
```

---

# Technology Stack

### Machine Learning

* Python
* NumPy
* Pandas
* Scikit-learn
* TensorFlow
* Keras
* Optuna

### Deep Learning

* Convolutional Neural Networks
* EEGNet-inspired architecture
* Depthwise convolution
* Separable convolution
* Batch normalization
* Dropout
* Global average pooling
* AdamW optimization

### Explainable AI

* SHAP
* Feature importance analysis
* SHAP-based feature selection

### Data Processing

* SciPy
* RobustScaler
* NumPy preprocessing
* Feature normalization

### Visualization

* Matplotlib
* Seaborn

### Environment

* Jupyter Notebook
* Kaggle

---

# Project Structure

A typical repository can be organized as:

```text
EEG-FMRI/
│
├── EEG+FMRI.ipynb
├── README.md
│
├── models/
│   ├── best_eeg_model.h5
│   └── best_fmri_model.h5
│
├── results/
│   ├── classification_report
│   ├── confusion_matrix
│   └── shap_analysis
│
└── requirements.txt
```

The exact repository structure may differ depending on which generated models, plots, and datasets are committed.

---

# Reproducibility

The experiment uses fixed seeds:

```python
np.random.seed(42)
tf.random.set_seed(42)
```

However, exact reproducibility may still depend on the TensorFlow version, hardware, CUDA configuration, and other sources of nondeterminism in GPU-based training.

The dataset itself is not included in the repository and must be obtained separately.

---

# Methodological Considerations

Several aspects of the current implementation are important when interpreting the reported results.

### 1. Preprocessing and Cross-Validation

Some preprocessing operations are performed before the cross-validation loop. In a strict evaluation protocol, transformations such as feature scaling should be fitted exclusively on each training fold and then applied to that fold's validation data.

A future version should move fold-dependent preprocessing inside the CV pipeline.

### 2. fMRI Representation

The fMRI input is represented as a resized 2D feature map rather than the original 3D fMRI volume.

Consequently, the CNN is learning from the provided feature representation rather than directly modeling full spatial 3D neuroimaging structure.

### 3. Sample Alignment

EEG and fMRI samples are aligned by truncating the modalities and labels to the smallest available sample count.

This assumes that corresponding rows represent the same samples.

For a more robust multimodal pipeline, explicit subject/sample identifiers should be used for alignment.

### 4. Subject-Level Splitting

If multiple observations originate from the same subject, subject-level cross-validation would be preferable to prevent information from the same subject appearing in both training and validation folds.

### 5. SHAP Feature Selection

The improvement from:

```text
0.7816 → 0.8028
```

is based on cross-validation F1 scores. It should not be interpreted as proof of improved generalization without evaluation on an independent held-out dataset.

---

# What This Project Demonstrates

This project demonstrates practical experience across several parts of a modern ML workflow:

* Multimodal machine learning
* EEG signal processing
* fMRI feature processing
* Deep learning with TensorFlow/Keras
* Custom CNN architecture design
* Representation learning
* Deep feature extraction
* Multimodal feature fusion
* Hyperparameter optimization with Optuna
* Classical ML with Random Forest
* Cross-validation
* Handling class imbalance
* Regularization and training callbacks
* Explainable AI with SHAP
* Feature selection
* Model retraining and comparison
* Experimental evaluation
* Reproducible ML experimentation

The combination of neural representation learning with a classical ensemble classifier also provides a useful example of **hybrid ML architecture design** rather than relying on a single end-to-end model.

---

# Future Improvements

Potential extensions include:

* Subject-level cross-validation
* Strict fold-wise preprocessing
* Independent held-out test evaluation
* Explicit EEG/fMRI sample-ID alignment
* Direct 3D fMRI CNN or 3D feature extraction
* Transformer-based EEG representations
* More systematic multimodal fusion strategies
* Early fusion vs. late fusion comparisons
* Learned fusion weights
* Nested cross-validation for hyperparameter selection
* Additional explainability methods
* Stability analysis of SHAP-selected features
* Statistical comparison across repeated CV runs
* External validation on an independent dataset

---

# Disclaimer

This is a machine learning research/engineering project and is **not a clinical diagnostic system**. The reported metrics describe performance within the implemented dataset and experimental evaluation procedure and should not be interpreted as clinical validation.

---

# Author

**Tishys**

This repository was developed as a practical exploration of multimodal deep learning, representation learning, feature fusion, optimization, and explainable AI.
