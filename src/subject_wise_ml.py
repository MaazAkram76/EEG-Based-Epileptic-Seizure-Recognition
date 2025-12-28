import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import welch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Create results directory
os.makedirs('../results', exist_ok=True)


FILENAME = '../data.csv'

print("=" * 70)
print("SUBJECT-WISE FEATURE EXTRACTION & ML CLASSIFICATION")
print("=" * 70)

print(f"\nLoading {FILENAME}...")
df = pd.read_csv(FILENAME)
print(f"   Loaded dataset with shape: {df.shape}")

# Extract components
ids = df.iloc[:, 0]
X_raw = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

print(f"   Total samples: {len(y)}")
print(f"   Features per sample: {X_raw.shape[1]}")


def parse_subject_id(uid):
    """Extract suffix from ID (e.g., X21.V1.791 -> 791)"""
    try:
        parts = str(uid).split('.')
        last = parts[-1]
        if last.startswith('V'):
            last = last[1:]
        return int(last)
    except:
        return -1

print("\nParsing Subject IDs...")
subject_suffixes = ids.apply(parse_subject_id)

subject_ids = y * 10000 + subject_suffixes

print(f"   Identified {len(np.unique(subject_ids))} unique subjects")

# Create DataFrame with subject grouping
df_with_subjects = pd.DataFrame({
    'subject_id': subject_ids,
    'label': y
})

# Add all time-series features
for i in range(X_raw.shape[1]):
    df_with_subjects[f'X{i+1}'] = X_raw[:, i]


print("\nExtracting Subject-Wise Features...")
print("   Features to extract:")
print("   Statistical: Mean, Std, Min, Max, Median, Skewness, Kurtosis")
print("   Frequency: Delta, Theta, Alpha, Beta, Gamma band powers")

fs = 173.6  # Sampling frequency

def extract_subject_features(subject_data):
    """
    Extract optimized feature set for a single subject.
    Input: DataFrame with all time segments for one subject
    Output: Dictionary of aggregated features
    """
    features = {}
    
    # Get all time-series columns (X1 to X178)
    time_series_cols = [col for col in subject_data.columns if col.startswith('X')]
    
    # Extract raw signals for this subject (all segments)
    raw_signals = subject_data[time_series_cols].values  # Shape: (num_segments, 178)
    

    scaler_local = StandardScaler()
    scaled_signals = scaler_local.fit_transform(raw_signals.T).T  # Scale across time points
    
    # Aggregate all segments into one long time series
    all_values_raw = raw_signals.flatten()
    

    features['mean'] = np.mean(all_values_raw)
    features['std'] = np.std(all_values_raw)
    features['min'] = np.min(all_values_raw)
    features['max'] = np.max(all_values_raw)
    features['median'] = np.median(all_values_raw)
    features['iqr'] = np.percentile(all_values_raw, 75) - np.percentile(all_values_raw, 25)
    features['skewness'] = stats.skew(all_values_raw)
    features['kurtosis'] = stats.kurtosis(all_values_raw)
    

    
    band_powers = {'Delta': [], 'Theta': [], 'Alpha': [], 'Beta': [], 'Gamma': []}
    
    # Define frequency bands
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 50)
    }
    
    # Process each segment (using SCALED data)
    for segment_idx in range(scaled_signals.shape[0]):
        signal_scaled = scaled_signals[segment_idx, :]
        
        # Calculate PSD using Welch's method
        f, psd = welch(signal_scaled, fs, nperseg=178)
        
        # Extract power in each frequency band
        for band_name, (low, high) in bands.items():
            idx_band = np.logical_and(f >= low, f <= high)
            if np.sum(idx_band) == 0:
                band_powers[band_name].append(0.0)
            else:
                # Integrate using trapezoidal rule
                power = np.trapz(psd[idx_band], f[idx_band])
                band_powers[band_name].append(power)
    
    # Aggregate band powers: mean and std only (2 per band = 10 features)
    for band_name in bands.keys():
        features[f'{band_name}_mean'] = np.mean(band_powers[band_name])
        features[f'{band_name}_std'] = np.std(band_powers[band_name])
    


    features['num_segments'] = len(subject_data)
    

    zero_crossings = []
    for segment_idx in range(raw_signals.shape[0]):
        signal = raw_signals[segment_idx, :]
        zc = np.sum(np.diff(np.sign(signal)) != 0)
        zero_crossings.append(zc)
    features['zero_crossing_rate'] = np.mean(zero_crossings)
    

    energies = []
    for segment_idx in range(raw_signals.shape[0]):
        signal = raw_signals[segment_idx, :]
        energy = np.sum(signal**2)
        energies.append(energy)
    features['energy_mean'] = np.mean(energies)
    features['energy_std'] = np.std(energies)
    
    return features

# Group by subject and extract features
print("   Processing subjects...")
subject_features_list = []

for subject_id in df_with_subjects['subject_id'].unique():
    subject_data = df_with_subjects[df_with_subjects['subject_id'] == subject_id]
    
    # Extract features
    features = extract_subject_features(subject_data)
    
    # Add subject ID and label
    features['subject_id'] = subject_id
    features['label'] = subject_data['label'].iloc[0]  # All segments have same label
    
    subject_features_list.append(features)

# Create feature DataFrame
df_features = pd.DataFrame(subject_features_list)

print(f"   Extracted {len(df_features.columns)-2} features per subject")
print(f"   Total subjects: {len(df_features)}")

# Save features
output_file = 'subject_wise_features.csv'
df_features.to_csv(output_file, index=False)
print(f"\nSaved features to '{output_file}'")


print("\nPreparing Data for Machine Learning...")

# Separate features and labels
X = df_features.drop(['subject_id', 'label'], axis=1).values
y = df_features['label'].values
groups = df_features['subject_id'].values

print(f"   Feature matrix shape: {X.shape}")
print(f"   Label distribution:")
for label in sorted(np.unique(y)):
    count = np.sum(y == label)
    print(f"      Class {label}: {count} subjects ({count/len(y)*100:.1f}%)")


print("\nPerforming Subject-Aware Split...")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"   Train set: {len(X_train)} subjects")
print(f"   Test set:  {len(X_test)} subjects")

# Verify no subject overlap
train_subjects = set(groups[train_idx])
test_subjects = set(groups[test_idx])
overlap = train_subjects.intersection(test_subjects)
print(f"   Subject overlap: {len(overlap)} (should be 0)")


print("\nScaling Features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   Features standardized")


print("\n" + "=" * 70)
print("TRAINING MACHINE LEARNING MODELS")
print("=" * 70)

# Define models (3 core classifiers)
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

results = {}

for model_name, model in models.items():
    print(f"\n{'-' * 70}")
    print(f"Training: {model_name}")
    print(f"{'-' * 70}")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Evaluate
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"   Training Accuracy:   {train_acc*100:.2f}%")
    print(f"   Test Accuracy:       {test_acc*100:.2f}%")
    
    # Store results
    results[model_name] = {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'y_pred': y_pred_test
    }


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

# Sort by test accuracy
sorted_results = sorted(results.items(), key=lambda x: x[1]['test_acc'], reverse=True)

print("\nModel Performance Summary:")
print(f"{'Model':<25} {'Train Acc':<12} {'Test Acc':<12} {'Overfitting':<12}")
print("-" * 70)

for model_name, res in sorted_results:
    overfit = res['train_acc'] - res['test_acc']
    print(f"{model_name:<25} {res['train_acc']*100:>10.2f}%  {res['test_acc']*100:>10.2f}%  {overfit*100:>10.2f}%")

# Best model
best_model_name = sorted_results[0][0]
best_result = sorted_results[0][1]

print("\n" + "=" * 70)
print(f"BEST MODEL: {best_model_name}")
print("=" * 70)

# Class names
target_names = [
    'Seizure (1)',
    'Tumor Area (2)',
    'Healthy Area (3)',
    'Eyes Closed (4)',
    'Eyes Open (5)'
]

print("\nDetailed Classification Report:")
print(classification_report(y_test, best_result['y_pred'], target_names=target_names, digits=4))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, best_result['y_pred'])
print("\nPredicted Label")
print("Actual Label")
print(f"{'':>15}", end='')
for i in range(1, 6):
    print(f"Class {i:>3}", end='  ')
print()
for i, row in enumerate(cm):
    print(f"Class {i+1:>3}        ", end='')
    for val in row:
        print(f"{val:>7}", end='  ')
    print()


if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print("\n" + "=" * 70)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("=" * 70)
    
    feature_names = [col for col in df_features.columns if col not in ['subject_id', 'label']]
    importances = best_result['model'].feature_importances_
    
    # Sort by importance
    indices = np.argsort(importances)[::-1][:10]
    
    print(f"\n{'Rank':<6} {'Feature':<25} {'Importance':<12}")
    print("-" * 50)
    for rank, idx in enumerate(indices, 1):
        print(f"{rank:<6} {feature_names[idx]:<25} {importances[idx]:.6f}")


print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create figure with subplots
fig = plt.figure(figsize=(20, 12))


ax1 = plt.subplot(2, 3, 1)
cm_best = confusion_matrix(y_test, best_result['y_pred'])
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Seizure', 'Tumor', 'Healthy', 'Eyes Closed', 'Eyes Open'],
            yticklabels=['Seizure', 'Tumor', 'Healthy', 'Eyes Closed', 'Eyes Open'],
            cbar_kws={'label': 'Count'})
ax1.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
ax1.set_ylabel('True Label', fontsize=12)
ax1.set_xlabel('Predicted Label', fontsize=12)


ax2 = plt.subplot(2, 3, 2)
model_names = [name for name, _ in sorted_results]
train_accs = [res['train_acc']*100 for _, res in sorted_results]
test_accs = [res['test_acc']*100 for _, res in sorted_results]

x = np.arange(len(model_names))
width = 0.35

rects1 = ax2.bar(x - width/2, train_accs, width, label='Train Accuracy', alpha=0.8)
rects2 = ax2.bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)

ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Model Comparison: Train vs Test Accuracy', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(model_names, rotation=15, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for rect in rects1 + rects2:
    height = rect.get_height()
    ax2.annotate(f'{height:.1f}%',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)


ax3 = plt.subplot(2, 3, 3)
class_labels = ['Seizure\n(1)', 'Tumor\n(2)', 'Healthy\n(3)', 'Eyes Closed\n(4)', 'Eyes Open\n(5)']
class_counts = [np.sum(y == i) for i in range(1, 6)]
colors = sns.color_palette("husl", 5)

wedges, texts, autotexts = ax3.pie(class_counts, labels=class_labels, autopct='%1.1f%%',
                                     colors=colors, startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)
ax3.set_title('Class Distribution in Dataset', fontsize=14, fontweight='bold')


if best_model_name in ['Random Forest', 'Gradient Boosting']:
    ax4 = plt.subplot(2, 3, 4)
    feature_names = [col for col in df_features.columns if col not in ['subject_id', 'label']]
    importances = best_result['model'].feature_importances_
    
    # Get top 10 features
    indices = np.argsort(importances)[::-1][:10]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    y_pos = np.arange(len(top_features))
    ax4.barh(y_pos, top_importances, alpha=0.8, color='steelblue')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(top_features)
    ax4.invert_yaxis()
    ax4.set_xlabel('Importance', fontsize=12)
    ax4.set_title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
else:
    ax4 = plt.subplot(2, 3, 4)
    ax4.text(0.5, 0.5, f'Feature importance not available\nfor {best_model_name}',
             ha='center', va='center', fontsize=12, transform=ax4.transAxes)
    ax4.axis('off')


ax5 = plt.subplot(2, 3, 5)

f1_scores = []
precisions = []
recalls = []

for class_label in range(1, 6):
    y_test_binary = (y_test == class_label).astype(int)
    y_pred_binary = (best_result['y_pred'] == class_label).astype(int)
    
    f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)
    prec = precision_score(y_test_binary, y_pred_binary, zero_division=0)
    rec = recall_score(y_test_binary, y_pred_binary, zero_division=0)
    
    f1_scores.append(f1 * 100)
    precisions.append(prec * 100)
    recalls.append(rec * 100)

x = np.arange(5)
width = 0.25

rects1 = ax5.bar(x - width, precisions, width, label='Precision', alpha=0.8)
rects2 = ax5.bar(x, recalls, width, label='Recall', alpha=0.8)
rects3 = ax5.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)

ax5.set_ylabel('Score (%)', fontsize=12)
ax5.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'])
ax5.legend()
ax5.grid(axis='y', alpha=0.3)
ax5.set_ylim([0, 105])


ax6 = plt.subplot(2, 3, 6)
model_names_full = list(results.keys())
train_accs_full = [results[name]['train_acc']*100 for name in model_names_full]
test_accs_full = [results[name]['test_acc']*100 for name in model_names_full]
overfit_gaps = [train - test for train, test in zip(train_accs_full, test_accs_full)]

colors_overfit = ['green' if gap < 5 else 'orange' if gap < 10 else 'red' for gap in overfit_gaps]

y_pos = np.arange(len(model_names_full))
ax6.barh(y_pos, overfit_gaps, alpha=0.8, color=colors_overfit)
ax6.set_yticks(y_pos)
ax6.set_yticklabels(model_names_full)
ax6.invert_yaxis()
ax6.set_xlabel('Overfitting Gap (%)', fontsize=12)
ax6.set_title('Overfitting Analysis (Train - Test)', fontsize=14, fontweight='bold')
ax6.axvline(x=5, color='orange', linestyle='--', alpha=0.5, label='5% threshold')
ax6.axvline(x=10, color='red', linestyle='--', alpha=0.5, label='10% threshold')
ax6.legend(fontsize=9)
ax6.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('../results/ml_evaluation_results.png', dpi=300, bbox_inches='tight')
print("   Saved: ../results/ml_evaluation_results.png")
plt.close()


fig2, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (model_name, res) in enumerate(sorted_results):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['S', 'T', 'H', 'EC', 'EO'],
                yticklabels=['S', 'T', 'H', 'EC', 'EO'],
                cbar_kws={'label': 'Count'})
    axes[idx].set_title(f'{model_name}\nAccuracy: {res["test_acc"]*100:.2f}%', 
                       fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('True Label', fontsize=10)
    axes[idx].set_xlabel('Predicted Label', fontsize=10)

plt.tight_layout()
plt.savefig('../results/ml_all_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("   Saved: ../results/ml_all_confusion_matrices.png")
plt.close()

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOutput files:")
print(f"   -> {output_file}")
print(f"   -> ../results/ml_evaluation_results.png")
print(f"   -> ../results/ml_all_confusion_matrices.png")
print(f"\nBest Model: {best_model_name} with {best_result['test_acc']*100:.2f}% test accuracy")
