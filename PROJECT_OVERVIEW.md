# EEG Signal Classification: Project Overview

**Team:** 231192, 231216, 231232  
**Course:** Biomedical Data Analysis and Processing (BSAI-5A)  
**Institution:** Air University

---

## 📋 Table of Contents

1. [Project Goal](#project-goal)
2. [Dataset Overview](#dataset-overview)
3. [Approach Comparison](#approach-comparison)
4. [Machine Learning Pipeline](#machine-learning-pipeline)
5. [Deep Learning Pipeline](#deep-learning-pipeline)
6. [Results Comparison](#results-comparison)

---

## 🎯 Project Goal

**Objective:** Classify EEG signals into 5 categories representing different brain states:
- **Class 1:** Seizure Activity (epileptic)
- **Class 2:** Tumor Area Recording
- **Class 3:** Healthy Brain Area
- **Class 4:** Eyes Closed (healthy, resting)
- **Class 5:** Eyes Open (healthy, active)

**Research Question:** Can we effectively distinguish between pathological (seizure, tumor) and normal brain activity using automated classification?

---

## 📊 Dataset Overview

- **Source:** UCI Machine Learning Repository - Epileptic Seizure Recognition
- **Total Samples:** 11,500 EEG segments
- **Features per Sample:** 178 time-series values (1-second recordings at 173.6 Hz)
- **Classes:** 5 (balanced distribution, ~2,300 samples each)
- **Unique Subjects:** ~500 individuals

**Critical Challenge:** Subject-aware splitting to prevent data leakage (same subject should not appear in both train and test sets)

---

## 🔄 Approach Comparison

We implemented two fundamentally different approaches:

| Aspect | Machine Learning | Deep Learning |
|--------|-----------------|---------------|
| **Philosophy** | Manual feature engineering | End-to-end learning |
| **Input** | 22 hand-crafted features | Raw 178-point time series |
| **Model Complexity** | Simple (Random Forest, Logistic Regression, KNN) | Complex (Transformer with attention) |
| **Interpretability** | High (feature importance) | Low (black box) |
| **Training Time** | Fast (~seconds) | Moderate (~minutes) |
| **Best Accuracy** | 87.00% | 73.57% |

---

# 🤖 Machine Learning Pipeline

## 1. Preprocessing

### What?
Transform raw EEG time-series data into subject-level aggregated signals.

### How?
1. **Subject Grouping:**
   ```python
   subject_ids = y * 10000 + subject_suffixes
   ```
   - Combines class label and subject ID to ensure uniqueness
   - Prevents same subject from appearing in train and test sets

2. **Data Aggregation:**
   - Group all time segments belonging to the same subject
   - Each subject has multiple 1-second EEG recordings

### Why?
- **Prevent Data Leakage:** Ensures model generalizes to new subjects, not just new segments
- **Realistic Evaluation:** Mimics real-world scenario where we classify signals from unseen patients
- **Statistical Validity:** Train/test split respects subject boundaries

---

## 2. Feature Extraction

### What?
Extract 22 meaningful features from raw time-series data for each subject.

### How?

#### **Statistical Features (8 features)**
```python
features['mean'] = np.mean(all_values_raw)
features['std'] = np.std(all_values_raw)
features['skewness'] = stats.skew(all_values_raw)
features['kurtosis'] = stats.kurtosis(all_values_raw)
```

**What they capture:**
- **Mean/Std:** Signal amplitude and variability
- **Min/Max:** Dynamic range
- **Median/IQR:** Robust central tendency
- **Skewness:** Asymmetry (seizures often show asymmetric spikes)
- **Kurtosis:** Tail heaviness (abnormal events)

#### **Frequency Domain Features (10 features)**
```python
# Welch's method for Power Spectral Density
f, psd = welch(signal_scaled, fs=173.6, nperseg=178)

# Extract power in each frequency band
bands = {
    'Delta': (0.5, 4),    # Deep sleep
    'Theta': (4, 8),      # Drowsiness
    'Alpha': (8, 13),     # Relaxed wakefulness
    'Beta': (13, 30),     # Active thinking
    'Gamma': (30, 50)     # High-level cognition
}
```

**What they capture:**
- **Delta:** Slow waves (high in sleep, tumors)
- **Theta:** Meditation, creativity
- **Alpha:** Relaxed but alert (eyes closed)
- **Beta:** Active concentration (eyes open)
- **Gamma:** Cognitive processing

**Why frequency matters:**
- Seizures show abnormal frequency patterns
- Eyes open/closed have distinct alpha band signatures
- Tumors disrupt normal frequency distribution

#### **Temporal Features (3 features)**
```python
features['zero_crossing_rate'] = np.mean(zero_crossings)
features['num_segments'] = len(subject_data)
```

**What they capture:**
- **Zero Crossing Rate:** Signal oscillation frequency
- **Number of Segments:** Data availability per subject

#### **Energy Features (2 features)**
```python
energy = np.sum(signal**2)
features['energy_mean'] = np.mean(energies)
```

**What they capture:**
- Signal power/intensity
- Seizures often have higher energy

### Why Feature Engineering?
- **Domain Knowledge:** Leverages neuroscience understanding of brain signals
- **Dimensionality Reduction:** 178 → 22 features (8x compression)
- **Interpretability:** Can explain why model makes decisions
- **Efficiency:** Faster training with fewer features

---

## 3. Model Training

### What?
Train three classical ML algorithms on extracted features.

### How?

#### **Models Trained:**

1. **Random Forest (Best: 87.00% accuracy)**
   ```python
   RandomForestClassifier(n_estimators=100, random_state=42)
   ```
   - Ensemble of 100 decision trees
   - Votes on final classification
   - Handles non-linear relationships

2. **Logistic Regression**
   ```python
   LogisticRegression(max_iter=1000, random_state=42)
   ```
   - Linear decision boundaries
   - Fast, interpretable baseline

3. **K-Nearest Neighbors**
   ```python
   KNeighborsClassifier(n_neighbors=5)
   ```
   - Instance-based learning
   - Classifies based on 5 nearest neighbors

#### **Training Process:**
1. **Feature Scaling:**
   ```python
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   ```
   - Standardize features to zero mean, unit variance
   - Prevents features with large ranges from dominating

2. **Subject-Aware Split:**
   ```python
   GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
   ```
   - 80% subjects for training
   - 20% subjects for testing
   - Zero overlap guaranteed

### Why These Models?
- **Random Forest:** Robust, handles feature interactions, provides importance scores
- **Logistic Regression:** Simple baseline, fast inference
- **KNN:** Non-parametric, captures local patterns

---

## 4. Evaluation and Results

### What?
Comprehensive performance analysis across multiple metrics.

### How?

#### **Metrics Computed:**

1. **Accuracy:** Overall correctness
   ```
   Random Forest: 87.00%
   Logistic Regression: ~75%
   KNN: ~70%
   ```

2. **Per-Class Performance:**
   - **Precision:** Of predicted class X, how many are actually X?
   - **Recall:** Of actual class X, how many did we find?
   - **F1-Score:** Harmonic mean of precision and recall

3. **Confusion Matrix:** Shows which classes are confused
   ```
   Example: Tumor vs Healthy confusion indicates similar patterns
   ```

4. **Feature Importance (Random Forest):**
   ```
   Top Features:
   1. Alpha_mean (0.12) - Eyes open/closed discrimination
   2. Beta_mean (0.10) - Active vs resting states
   3. Gamma_std (0.08) - Seizure detection
   ```

#### **Visualizations Generated:**
- 6-panel evaluation dashboard
- Model comparison charts
- Confusion matrices for all models
- Overfitting analysis

### Why These Metrics?
- **Accuracy:** Overall performance indicator
- **Per-Class Metrics:** Identify which brain states are harder to classify
- **Confusion Matrix:** Understand misclassification patterns
- **Feature Importance:** Validate neuroscience hypotheses

---

# 🧠 Deep Learning Pipeline

## 1. Preprocessing

### What?
Prepare raw time-series data for neural network input.

### How?

1. **Subject-Aware Splitting:**
   ```python
   groups = y * 10000 + raw_indices
   GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
   ```
   - Same strategy as ML approach
   - Ensures subject disjointness

2. **Feature Scaling:**
   ```python
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   ```
   - Normalize to zero mean, unit variance
   - Fit only on training data to prevent leakage

3. **Label Adjustment:**
   ```python
   y_train = y_train - 1  # Convert 1-5 to 0-4
   ```
   - PyTorch CrossEntropyLoss expects 0-indexed labels

### Why?
- **Scaling:** Neural networks converge faster with normalized inputs
- **Subject Splitting:** Same rationale as ML (prevent overfitting to subjects)
- **Label Indexing:** Technical requirement for PyTorch

---

## 2. Feature Extraction

### What?
**None!** The model learns features automatically.

### How?
Instead of manual feature engineering, the Transformer architecture learns:
- Temporal patterns through self-attention
- Frequency characteristics through learned filters
- Complex interactions between time points

### Why No Manual Features?
- **End-to-End Learning:** Model discovers optimal representations
- **Flexibility:** Can learn features we didn't think of
- **Generalization:** Not limited by human domain knowledge

**Trade-off:** Requires more data and computation, less interpretable

---

## 3. Model Training

### What?
Train a patch-based Transformer neural network on raw EEG signals.

### How?

#### **Model Architecture:**

```python
class EEGTransformer(nn.Module):
    def __init__(self, input_dim=178, num_classes=5, 
                 patch_size=20, d_model=64, nhead=4, num_layers=2):
```

**Components:**

1. **Patch Embedding:**
   ```python
   # Divide 178-point signal into 9 patches of size 20
   x = x.view(batch, num_patches=9, patch_size=20)
   x = self.patch_embedding(x)  # Linear projection to d_model=64
   ```
   
   **What:** Splits time series into overlapping windows
   **Why:** Reduces sequence length, captures local patterns

2. **Positional Encoding:**
   ```python
   self.pos_embedding = nn.Parameter(torch.randn(1, 9, 64))
   x = x + self.pos_embedding
   ```
   
   **What:** Adds position information to each patch
   **Why:** Transformer has no inherent notion of order

3. **Transformer Encoder:**
   ```python
   TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128)
   ```
   
   **What:** Multi-head self-attention mechanism
   **Why:** Learns relationships between different time patches
   
   **How it works:**
   - Each patch "attends" to all other patches
   - Learns which time points are important for classification
   - 4 attention heads capture different patterns simultaneously

4. **Global Average Pooling:**
   ```python
   x = x.mean(dim=1)  # Average across patches
   ```
   
   **What:** Aggregates patch representations
   **Why:** Creates fixed-size representation for classification

5. **Classification Head:**
   ```python
   self.fc = nn.Linear(64, 5)  # 64 features → 5 classes
   ```

#### **Training Configuration:**

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 15
batch_size = 64
```

**Training Loop:**
1. Forward pass: Compute predictions
2. Compute loss: Compare with true labels
3. Backward pass: Calculate gradients
4. Update weights: Adjust model parameters

**Training Progress:**
```
Epoch 1:  Loss: 1.18, Acc: 44.52%
Epoch 5:  Loss: 0.67, Acc: 67.28%
Epoch 10: Loss: 0.59, Acc: 71.42%
Epoch 15: Loss: 0.53, Acc: 74.67%
```

### Why This Architecture?
- **Transformers:** State-of-the-art for sequence modeling
- **Patch-based:** Reduces computational cost, captures local patterns
- **Self-Attention:** Learns long-range dependencies in time series
- **Lightweight:** Only 2 layers, 4 heads (trainable on CPU)

---

## 4. Evaluation and Results

### What?
Assess model performance on held-out test subjects.

### How?

#### **Test Set Evaluation:**
```python
model.eval()
with torch.no_grad():
    outputs = model(inputs)
    predictions = torch.max(outputs, 1)
```

**Final Results:**
- **Overall Accuracy:** 73.57%
- **Training Accuracy:** 74.67%
- **Overfitting:** Minimal (1.1% gap)

#### **Per-Class Performance:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Seizure (1) | 96.15% | 96.62% | 96.39% | 621 |
| Tumor (2) | 56.85% | 38.32% | 45.78% | 368 |
| Healthy (3) | 55.73% | 59.51% | 57.56% | 368 |
| Eyes Closed (4) | 70.72% | 80.31% | 75.21% | 391 |
| Eyes Open (5) | 70.73% | 75.72% | 73.14% | 552 |

**Key Observations:**
1. **Excellent Seizure Detection:** 96%+ across all metrics
   - Seizures have very distinct patterns
   - Most critical for medical diagnosis

2. **Struggle with Tumor/Healthy:** ~50-60% accuracy
   - Similar frequency characteristics
   - Requires more subtle pattern recognition

3. **Good Eyes Open/Closed:** ~70-75%
   - Clear alpha band differences
   - Easier discrimination

#### **Visualizations Generated:**
- Training loss/accuracy curves
- Confusion matrix
- Per-class performance bars
- Model architecture summary

### Why These Results?
- **Seizure Success:** Strong, unique signal characteristics
- **Tumor Difficulty:** Subtle differences from healthy tissue
- **Overall Performance:** Good but below ML approach
  - Reason: Limited training data for deep learning
  - Deep learning typically needs 10,000+ samples per class
  - We have only ~2,000 samples per class

---

# 📊 Results Comparison

## Performance Summary

| Metric | Machine Learning | Deep Learning |
|--------|-----------------|---------------|
| **Best Model** | Random Forest | Transformer |
| **Test Accuracy** | **87.00%** | 73.57% |
| **Training Time** | ~30 seconds | ~5 minutes |
| **Seizure Detection** | ~95% | **96%+** |
| **Tumor Detection** | ~80% | ~45% |
| **Interpretability** | High | Low |

## When to Use Each Approach?

### Use Machine Learning When:
✅ Limited data available (<10,000 samples)  
✅ Interpretability is critical (medical diagnosis)  
✅ Fast training/inference required  
✅ Domain expertise available for feature engineering  
✅ Computational resources limited  

### Use Deep Learning When:
✅ Large datasets available (>100,000 samples)  
✅ Complex, high-dimensional raw data  
✅ Patterns are too complex for manual feature design  
✅ Computational resources abundant (GPUs)  
✅ Transfer learning possible (pre-trained models)  

## Key Insights

1. **Feature Engineering Matters:** Hand-crafted features outperformed raw data with limited samples
2. **Domain Knowledge is Powerful:** Understanding brain frequency bands gave ML a significant advantage
3. **Data Efficiency:** ML achieved 87% with 22 features vs DL's 73% with 178 raw values
4. **Task-Specific Performance:** DL excelled at seizure detection, ML better at overall classification

---

# 🎓 Conclusions

## What We Learned

1. **Subject-Aware Splitting is Critical:**
   - Prevents inflated accuracy from data leakage
   - Ensures real-world generalization

2. **Feature Engineering Still Relevant:**
   - With limited data, domain knowledge beats brute-force learning
   - Frequency domain features are highly informative for EEG

3. **Model Selection Depends on Context:**
   - Random Forest: Best overall performance
   - Transformer: Best seizure detection
   - Choice depends on specific clinical requirements

4. **Visualization is Essential:**
   - Confusion matrices reveal class-specific strengths/weaknesses
   - Feature importance validates neuroscience hypotheses

## Future Improvements

1. **Data Augmentation:** Generate synthetic EEG samples
2. **Hybrid Approach:** Combine hand-crafted and learned features
3. **Ensemble Methods:** Combine ML and DL predictions
4. **Advanced Architectures:** Try CNNs, RNNs, or larger Transformers
5. **Transfer Learning:** Pre-train on larger EEG datasets

---

## 📚 References

- Dataset: UCI Machine Learning Repository - Epileptic Seizure Recognition
- Welch's Method: Power Spectral Density estimation
- Transformer Architecture: "Attention is All You Need" (Vaswani et al., 2017)
- EEG Frequency Bands: Clinical neurophysiology standards

---

**Project Repository:** `BDPA_BSAI5A_192_216_232`  
**Team Members:** Muhammad Maaz Akram (231192), Muhammad Abubakar (231216), Kamran Ahmed (231232)  
**Course Instructor:** Dr. Abdul Haleem Butt  
**Institution:** Air University, Department of Biomedical Engineering
