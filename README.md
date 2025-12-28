# EEG Signal Classification: Machine Learning vs Foundation Models

**Course:** Biomedical Data Analysis and Processing (BSAI-5A)  
**Team Members:** 192, 216, 232  
**Institution:** Air University

---

## 📋 Project Overview

This project implements and compares two approaches for EEG signal classification:

1. **Traditional Machine Learning** - Subject-wise feature extraction with classical ML algorithms
2. **Deep Learning Foundation Model** - Patch-based Transformer architecture for end-to-end learning

The goal is to classify EEG signals into 5 categories:
- **Class 1:** Seizure Activity
- **Class 2:** Tumor Area Recording
- **Class 3:** Healthy Brain Area
- **Class 4:** Eyes Closed
- **Class 5:** Eyes Open

---

## 📁 Repository Structure

```
BDPA_BSAI5A_192_216_232/
├── README.md                          # This file
├── PROJECT_OVERVIEW.md                # Detailed technical overview
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── docs/
│   ├── Project_Proposal.md            # Initial project proposal
│   ├── Project_Report.md              # Comprehensive project report
│   └── Dataset_Overview.txt           # Dataset description
│
├── src/
│   ├── subject_wise_ml.py             # ML pipeline with feature extraction
│   ├── eeg_foundation_model.py        # Transformer-based deep learning model
│   └── verify_split.py                # Data split verification utility
│
├── data/
│   ├── README.md                      # Dataset download instructions
│   ├── processed_eeg_features.csv     # Extracted EEG features
│   └── subject_wise_features.csv      # Subject-aggregated features
│
└── results/
    ├── ml_evaluation_results.png      # ML 6-panel evaluation dashboard
    ├── ml_all_confusion_matrices.png  # ML confusion matrices comparison
    ├── dl_evaluation_results.png      # DL 6-panel evaluation dashboard
    └── dl_confusion_matrix_detailed.png  # DL detailed confusion matrix
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for deep learning model

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd BDPA_BSAI5A_192_216_232
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   
   The original EEG dataset should be placed in the project root as `data.csv`. 
   
   **Dataset Source:** [Epileptic Seizure Recognition Dataset](https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition)
   
   > **Note:** Due to size constraints, the raw dataset (`data.csv`) is not included in this repository. Please download it from the source above.

---

## 💻 Usage

### 1. Traditional Machine Learning Approach

This script extracts subject-wise features and trains multiple ML classifiers:

```bash
cd src
python subject_wise_ml.py
```

**What it does:**
- Loads raw EEG time-series data
- Performs subject-aware data splitting (no subject overlap between train/test)
- Extracts 23-25 features per subject:
  - Statistical features (mean, std, skewness, kurtosis, etc.)
  - Frequency band powers (Delta, Theta, Alpha, Beta, Gamma)
  - Temporal features (zero-crossing rate, segment count)
  - Energy features
- Trains and evaluates:
  - Random Forest
  - Logistic Regression
  - K-Nearest Neighbors
- Outputs detailed classification reports and feature importance

**Output:**
- `subject_wise_features.csv` - Extracted features for all subjects
- `results/ml_evaluation_results.png` - 6-panel evaluation dashboard
- `results/ml_all_confusion_matrices.png` - Confusion matrices for all models
- Console output with model performance metrics

---

### 2. Deep Learning Foundation Model

This script implements a patch-based Transformer for end-to-end EEG classification:

```bash
cd src
python eeg_foundation_model.py
```

**What it does:**
- Loads raw EEG time-series data (178 time points per sample)
- Performs subject-aware data splitting
- Implements a Transformer architecture:
  - Patch embedding (divides signal into patches)
  - Positional encoding
  - Multi-head self-attention layers
  - Global average pooling
  - Classification head
- Trains for 15 epochs using Adam optimizer
- Evaluates on held-out test set

**Model Architecture:**
- Input: 178-dimensional EEG signal
- Patch size: 20
- Embedding dimension: 64
- Attention heads: 4
- Transformer layers: 2
- Output: 5 classes

**Output:**
- `results/dl_evaluation_results.png` - 6-panel evaluation dashboard
- `results/dl_confusion_matrix_detailed.png` - Detailed confusion matrix
- Console output with training progress and metrics

---

### 3. Data Verification

Verify the subject-aware split to ensure no data leakage:

```bash
cd src
python verify_split.py
```

---

## 📊 Results Summary

Detailed results are available in [`docs/Project_Report.md`](docs/Project_Report.md) and [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md).

**Visual Results:** All evaluation graphs are available in the [`results/`](results/) folder.

### Machine Learning Performance
- **Best Model:** Random Forest
- **Test Accuracy:** 87.00%
- **Key Features:** Frequency band powers (Alpha, Beta) and statistical moments
- **Visualizations:** 6-panel dashboard, confusion matrices, feature importance

### Deep Learning Performance
- **Model:** Patch-based Transformer
- **Test Accuracy:** 73.57%
- **Seizure Detection:** 96%+ (excellent)
- **Advantages:** End-to-end learning, no manual feature engineering
- **Visualizations:** Training curves, confusion matrix, per-class performance

---

## 🔬 Key Features

### Subject-Aware Splitting
Both approaches ensure **zero subject overlap** between training and test sets to prevent data leakage and ensure generalization.

### Feature Engineering (ML Approach)
- **Statistical Features:** Capture signal distribution characteristics
- **Frequency Domain:** Welch's method for power spectral density
- **Temporal Features:** Zero-crossing rate, segment variability

### Foundation Model (DL Approach)
- **Patch-based Processing:** Divides time series into patches
- **Self-Attention:** Learns temporal dependencies
- **Positional Encoding:** Preserves temporal order

---

## 📚 Documentation

- **[Project Overview](PROJECT_OVERVIEW.md)** - Detailed technical explanation of preprocessing, feature extraction, training, and evaluation
- **[Project Proposal](docs/Project_Proposal.md)** - Initial project plan and objectives
- **[Project Report](docs/Project_Report.md)** - Comprehensive analysis, methodology, and results
- **[Dataset Overview](docs/Dataset_Overview.txt)** - Dataset description and preprocessing notes

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **Machine Learning:** scikit-learn, pandas, numpy, scipy
- **Deep Learning:** PyTorch
- **Visualization:** matplotlib, seaborn
- **Data Processing:** StandardScaler, GroupShuffleSplit

---

## 📝 Citation

If you use this code or methodology, please cite:

```
EEG Signal Classification: Machine Learning vs Foundation Models
Team: 192, 216, 232
Course: Biomedical Data Analysis and Processing (BSAI-5A)
Air University, 2025
```

---

## 👥 Team Members

- **231192** - Muhammad Maaz Akram
- **231216** - Muhammad Abubakar
- **231232** - Kamran Ahmed

---

## 📄 License

This project is submitted as part of academic coursework at Air University.

---

## 🙏 Acknowledgments

- Dataset: UCI Machine Learning Repository - Epileptic Seizure Recognition
- Course Instructor: Dr. Abdul Haleem Butt
- Air University, Department of Biomedical Engineering

---

## 📧 Contact

For questions or clarifications, please contact the team members through the university portal.
