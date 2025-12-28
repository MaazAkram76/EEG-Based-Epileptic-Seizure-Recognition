# EEG-Based Epileptic Seizure Recognition Using Transformer Architecture
## Project Proposal

---

## 1. Dataset

### Dataset Overview
- **Source**: UCI Machine Learning Repository - Epileptic Seizure Recognition Data Set
- **File**: `data.csv`
- **Total Samples**: 11,500 EEG recordings
- **Features**: 178 time-series data points per sample
- **Duration**: Each sample represents 1 second of brain activity
- **Sampling Rate**: 173.6 Hz

### Data Structure
The dataset contains EEG recordings from 500 unique subjects, where each subject's 23.6-second recording was divided into 23 one-second segments (chunks). Each segment contains 178 sequential EEG amplitude measurements.

**Dataset Dimensions**:
- **Rows**: 11,500 (500 subjects × 23 segments each)
- **Columns**: 180 total
  - Column 0: Subject ID (format: `X[prefix].V[version].[suffix]`)
  - Columns 1-178: EEG signal values (X1, X2, ..., X178)
  - Column 179: Class label (y)

### Class Distribution
The dataset contains 5 classes representing different brain states:

| Class | Description | Clinical Significance |
|-------|-------------|----------------------|
| **1** | Seizure Activity | EEG recorded during epileptic seizure episode |
| **2** | Tumor Area | EEG from brain region where tumor is located |
| **3** | Healthy Area | EEG from healthy brain tissue (tumor patients) |
| **4** | Eyes Closed | EEG from healthy subjects with eyes closed |
| **5** | Eyes Open | EEG from healthy subjects with eyes open |

**Binary Classification Context**: Classes 2-5 represent non-seizure states, while Class 1 represents seizure activity. This enables both multi-class (5-way) and binary (seizure vs. non-seizure) classification tasks.

### Subject-Aware Data Splitting
**Critical Consideration**: To prevent data leakage, the dataset must be split at the **subject level**, not the sample level.

**Grouping Strategy**:
- Each subject has multiple time segments (up to 23 per subject)
- Subject identification: `GroupID = Class × 10,000 + Suffix`
- This ensures that all segments from the same subject remain in either training or testing set
- **Unique Subjects**: ~500 individuals
- **Split Ratio**: 80% training, 20% testing (subject-disjoint)

**Why This Matters**: Without subject-aware splitting, the model could memorize subject-specific patterns rather than learning generalizable seizure detection features, leading to inflated performance metrics that don't reflect real-world performance.

---

## 2. Preprocessing

### What: Preprocessing Steps

#### 2.1 Data Cleaning
- **Missing Value Handling**: Check for and impute any missing values using mean imputation
- **Data Type Conversion**: Convert features to `float32` and labels to `int64` for computational efficiency

#### 2.2 Standardization (Z-Score Normalization)
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Process**:
- Compute mean (μ) and standard deviation (σ) from **training set only**
- Transform each feature: `z = (x - μ) / σ`
- Apply same transformation parameters to test set

#### 2.3 Label Adjustment
- Convert labels from 1-based (1-5) to 0-based (0-4) for PyTorch compatibility
- `y_adjusted = y_original - 1`

### Why: Rationale for Preprocessing

#### Why Standardization?
1. **Scale Normalization**: EEG signals can have varying amplitudes across subjects due to:
   - Individual physiological differences
   - Electrode placement variations
   - Recording equipment calibration

2. **Neural Network Optimization**: 
   - Prevents features with larger scales from dominating the learning process
   - Enables faster convergence during gradient descent
   - Reduces risk of vanishing/exploding gradients

3. **Transformer Architecture Requirement**:
   - Attention mechanisms are sensitive to input scale
   - Standardized inputs improve stability of self-attention computations

#### Why Train-Only Statistics?
- **Prevents Data Leakage**: Using test set statistics would give the model indirect information about test data
- **Simulates Real Deployment**: In production, we only have access to training data statistics

#### Why Float32?
- **Memory Efficiency**: Reduces memory footprint by 50% compared to float64
- **GPU Acceleration**: Modern GPUs are optimized for float32 operations
- **Sufficient Precision**: EEG analysis doesn't require double precision

---

## 3. Feature Extraction

### What: Two Approaches Implemented

#### Approach 1: Frequency Band Power Features (`gemini1.py`)

**Extracted Features**:
Using Welch's method for Power Spectral Density (PSD) estimation, we extract power in 5 clinically relevant frequency bands:

| Band | Frequency Range | Clinical Association |
|------|----------------|---------------------|
| **Delta** | 0.5 - 4 Hz | Deep sleep, brain lesions |
| **Theta** | 4 - 8 Hz | Drowsiness, meditation, seizure activity |
| **Alpha** | 8 - 13 Hz | Relaxed wakefulness, eyes closed |
| **Beta** | 13 - 30 Hz | Active thinking, focus, anxiety |
| **Gamma** | 30 - 50 Hz | Cognitive processing, seizures |

**Implementation**:
```python
f, psd = welch(signal_row, fs=173.6, nperseg=178)
# Integrate power in each band using trapezoidal rule
power_delta = np.trapz(psd[delta_idx], f[delta_idx])
```

**Output**: 5 features per sample (Delta, Theta, Alpha, Beta, Gamma power)

#### Approach 2: Raw Time-Series with Patch Embedding (`eeg_foundation_model.py`)

**Patchification Strategy**:
- **Patch Size**: 20 time points
- **Input Dimension**: 178 → Padded to 180 (divisible by 20)
- **Number of Patches**: 9 patches per sample
- **Patch Embedding**: Linear projection from 20D → 64D (d_model)

**Positional Encoding**:
- Learnable positional embeddings (1, 9, 64)
- Added to patch embeddings to preserve temporal order

### Why: Rationale for Feature Extraction

#### Why Frequency Band Features?

1. **Clinical Relevance**:
   - Neurologists analyze EEG in frequency domain
   - Seizures show characteristic patterns in specific bands (especially Theta and Gamma)
   - Reduces dimensionality from 178 → 5 features while retaining diagnostic information

2. **Noise Reduction**:
   - Frequency analysis filters out high-frequency noise
   - Welch's method provides robust PSD estimates

3. **Interpretability**:
   - Band powers have direct clinical meaning
   - Easier to explain model decisions to medical professionals

#### Why Patch-Based Approach (Foundation Model)?

1. **Mimics Vision Transformers**:
   - Inspired by ViT (Vision Transformer) success
   - Treats time-series as sequence of "patches" instead of individual time points
   - Reduces sequence length (178 → 9), making self-attention computationally feasible

2. **Learns Hierarchical Features**:
   - Transformer learns its own feature representations
   - No manual feature engineering required
   - Can capture complex temporal patterns

3. **Scalability**:
   - Foundation model approach generalizes to other EEG tasks
   - Pre-training potential on larger unlabeled EEG datasets

4. **State-of-the-Art Technique**:
   - Aligns with modern deep learning best practices
   - Transformer architectures dominate NLP, vision, and increasingly biosignal analysis

#### Why Both Approaches?

- **Comparison**: Evaluate traditional signal processing vs. deep learning
- **Complementary Strengths**: Frequency features are interpretable; transformers are powerful
- **Research Value**: Demonstrates understanding of both classical and modern techniques

---

## 4. Model Training

### What: Transformer-Based Architecture

#### Model Architecture (`EEGTransformer`)

```
Input (178 time points)
    ↓
Padding (178 → 180)
    ↓
Patchify (180 → 9 patches × 20 points)
    ↓
Patch Embedding (20 → 64 dimensions)
    ↓
+ Positional Encoding (learnable)
    ↓
Transformer Encoder (2 layers, 4 heads)
    ↓
Global Average Pooling (9 patches → 1 vector)
    ↓
Classification Head (64 → 5 classes)
```

#### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Patch Size** | 20 | Time points per patch |
| **d_model** | 64 | Embedding dimension |
| **nhead** | 4 | Number of attention heads |
| **num_layers** | 2 | Transformer encoder layers |
| **dim_feedforward** | 128 | Hidden layer size in FFN |
| **Batch Size** | 64 | Samples per training batch |
| **Learning Rate** | 0.001 | Adam optimizer step size |
| **Epochs** | 15 | Training iterations |

#### Training Configuration

**Loss Function**: Cross-Entropy Loss
```python
criterion = nn.CrossEntropyLoss()
```

**Optimizer**: Adam
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**Device**: GPU (CUDA) if available, else CPU

### Why: Rationale for Model Design

#### Why Transformer Architecture?

1. **Self-Attention Mechanism**:
   - Captures long-range dependencies in time-series
   - Each patch can attend to all other patches
   - Learns which temporal regions are most relevant for classification

2. **Parallel Processing**:
   - Unlike RNNs/LSTMs, transformers process entire sequence in parallel
   - Faster training on modern GPUs
   - No vanishing gradient issues from sequential processing

3. **Foundation Model Paradigm**:
   - Transformers are the backbone of foundation models (BERT, GPT, ViT)
   - Demonstrates application of cutting-edge AI to biomedical signals
   - Potential for transfer learning from larger EEG datasets

#### Why These Hyperparameters?

**Patch Size = 20**:
- Balances local temporal detail with computational efficiency
- 178 ÷ 20 ≈ 9 patches (manageable sequence length)
- Each patch covers ~0.115 seconds (20 / 173.6 Hz)

**d_model = 64**:
- Sufficient capacity for 5-class problem
- Prevents overfitting on 11,500 samples
- Computationally efficient

**nhead = 4**:
- Must divide d_model evenly (64 ÷ 4 = 16 dimensions per head)
- Multiple heads learn different attention patterns
- Not too many (would overfit) or too few (would underfit)

**num_layers = 2**:
- Shallow architecture appropriate for dataset size
- Deeper models risk overfitting without more data
- Still captures hierarchical feature learning

**Learning Rate = 0.001**:
- Standard Adam learning rate
- Balances convergence speed and stability

**Batch Size = 64**:
- Fits in GPU memory
- Provides stable gradient estimates
- Enables ~143 batches per epoch (9,200 train samples ÷ 64)

#### Why Cross-Entropy Loss?

- Standard for multi-class classification
- Penalizes confident wrong predictions more heavily
- Outputs calibrated probabilities via softmax

#### Why Adam Optimizer?

- Adaptive learning rates per parameter
- Combines benefits of momentum and RMSprop
- Robust to hyperparameter choices
- Industry standard for deep learning

#### Why Global Average Pooling?

- Aggregates information from all 9 patches
- More robust than using only [CLS] token (common in NLP)
- Reduces overfitting compared to using all patch embeddings

---

## 5. Evaluation

### What: Evaluation Methodology

#### Metrics

1. **Classification Report** (per-class metrics):
   - **Precision**: Of predicted seizures, how many were correct?
   - **Recall**: Of actual seizures, how many did we detect?
   - **F1-Score**: Harmonic mean of precision and recall
   - **Support**: Number of samples per class

2. **Overall Accuracy**: Percentage of correct predictions across all classes

3. **Confusion Matrix** (implicit in classification report):
   - Shows which classes are confused with each other
   - Critical for medical applications (false negatives in seizure detection are dangerous)

#### Evaluation Protocol

```python
model.eval()  # Disable dropout/batch norm training behavior
with torch.no_grad():  # Disable gradient computation
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predictions = torch.max(outputs, dim=1)
```

**Key Aspects**:
- **Subject-Disjoint Test Set**: No overlap between training and test subjects
- **Zero-Shot Evaluation**: Model never sees test subjects during training
- **Batch Processing**: Efficient evaluation on GPU

#### Target Classes

```python
target_names = [
    'Seizure (1)',
    'Tumor Area (2)', 
    'Healthy Area (3)',
    'Eyes Closed (4)',
    'Eyes Open (5)'
]
```

### Why: Rationale for Evaluation Approach

#### Why Subject-Disjoint Evaluation?

1. **Realistic Performance Estimate**:
   - Simulates real clinical deployment
   - Model must generalize to new patients, not just new time segments

2. **Prevents Overfitting Detection**:
   - If model memorizes subject-specific patterns, performance will drop significantly
   - Validates that learned features are disease-related, not subject-specific

3. **Clinical Validity**:
   - In practice, model will diagnose patients it has never seen
   - Subject-aware splitting is gold standard in medical ML

#### Why Classification Report Over Just Accuracy?

1. **Class Imbalance Awareness**:
   - Dataset may have unequal class distribution
   - Accuracy can be misleading (e.g., 90% accuracy by always predicting majority class)

2. **Medical Decision Making**:
   - **High Recall for Seizure Class**: Critical to catch all seizures (minimize false negatives)
   - **High Precision**: Reduces false alarms (patient anxiety, unnecessary treatment)
   - F1-score balances both concerns

3. **Per-Class Insights**:
   - Identifies which brain states are harder to classify
   - Guides future model improvements

#### Why Precision and Recall Matter in Medical Context?

**Seizure Detection (Class 1)**:
- **High Recall Priority**: Missing a seizure (false negative) could be life-threatening
  - Patient might not receive timely intervention
  - Seizure could progress to status epilepticus
  
- **Precision Trade-off**: False positives (false alarms) are less critical but still problematic
  - Causes patient anxiety
  - May lead to unnecessary medication adjustments

**Ideal Scenario**: High recall (≥95%) with acceptable precision (≥85%)

#### Why Test on Entire Test Set?

- **Statistical Significance**: Larger test set provides more reliable performance estimates
- **Confidence Intervals**: More samples reduce variance in metrics
- **Rare Class Detection**: Ensures sufficient samples from minority classes

#### Why Disable Gradients During Evaluation?

```python
with torch.no_grad():
```

1. **Memory Efficiency**: Gradient computation requires storing intermediate activations
2. **Speed**: Inference is 2-3× faster without gradient tracking
3. **Correctness**: Prevents accidental model updates during evaluation

#### Why Report Multiple Metrics?

- **Accuracy**: Overall performance summary
- **Precision**: Confidence in positive predictions
- **Recall**: Coverage of actual positives
- **F1-Score**: Balanced metric when precision/recall trade-offs exist
- **Support**: Context for interpreting other metrics (low support = less reliable)

### Expected Outcomes

**Success Criteria**:
- Overall accuracy > 80% (demonstrates model learns meaningful patterns)
- Seizure class (Class 1) recall > 85% (critical for clinical utility)
- Performance significantly above random baseline (20% for 5-class problem)

**Comparison Baseline**:
- Random guessing: 20% accuracy
- Majority class baseline: ~20% (if balanced)
- Published benchmarks on this dataset: 85-95% (various methods)

---

## Summary

This project implements a **state-of-the-art Transformer-based architecture** for EEG-based epileptic seizure recognition, combining:

1. **Rigorous Data Handling**: Subject-aware splitting prevents data leakage
2. **Dual Feature Strategies**: Classical frequency analysis + modern patch embeddings
3. **Modern Architecture**: Transformer encoder with self-attention mechanisms
4. **Clinical Relevance**: Evaluation metrics aligned with medical decision-making priorities

The approach demonstrates both **traditional signal processing expertise** (frequency band extraction) and **cutting-edge deep learning** (foundation model paradigm), providing a comprehensive solution to automated seizure detection.
