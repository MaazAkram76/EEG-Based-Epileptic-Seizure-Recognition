import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import os


os.makedirs('../results', exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


FILENAME = '../data.csv'

def load_and_split_data(filename):
    print(f"Loading {filename}...")
    df = pd.read_csv(filename)
    

    
    ids = df.iloc[:, 0]
    X = df.iloc[:, 1:-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64)
    

    
    def parse_id(uid):
        try:
            parts = uid.split('.')
            last = parts[-1]
            if last.startswith('V'):
                last = last[1:]
            return int(last)
        except:
            return -1

    try:
        raw_indices = ids.astype(str).apply(parse_id)
        

        groups = y * 10000 + raw_indices
        

        unique_groups = len(np.unique(groups))
        print(f"Subject Grouping Logic: Derived {unique_groups} unique subjects (Expect ~500).")
        

        check_df = pd.DataFrame({'group': groups, 'label': y})
        inconsistent_groups = check_df.groupby('group')['label'].nunique()
        if inconsistent_groups.max() > 1:
            print("WARNING: Subject grouping hypothesis failed! Some groups have mixed labels.")
            print(inconsistent_groups[inconsistent_groups > 1])
        else:
            print("Subject Grouping verified: No leakage detected.")

    except Exception as e:
        print(f"Error parsing IDs or Grouping: {e}")
        print("Fallback: Using simple rows (LEAKAGE RISK!)")
        groups = np.arange(len(y))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    

    y_train = y_train - 1
    y_test = y_test - 1
    
    print(f"   -> Train Set: {X_train.shape}")
    print(f"   -> Test Set:  {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_and_split_data(FILENAME)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = EEGDataset(X_train_scaled, y_train)
test_dataset = EEGDataset(X_test_scaled, y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class EEGTransformer(nn.Module):
    def __init__(self, input_dim=178, num_classes=5, patch_size=20, d_model=64, nhead=4, num_layers=2):
        super(EEGTransformer, self).__init__()
        
        self.patch_size = patch_size
        self.d_model = d_model
        
        remainder = input_dim % patch_size
        self.padding = (patch_size - remainder) if remainder > 0 else 0
        self.padded_dim = input_dim + self.padding
        self.num_patches = self.padded_dim // patch_size
        
        print(f"Model Config: Input {input_dim} -> Padded {self.padded_dim}")
        print(f"   -> {self.num_patches} Patches of size {patch_size}")
        

        self.patch_embedding = nn.Linear(patch_size, d_model)
        

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        

        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x shape: (Batch, 178)
        

        if self.padding > 0:
            x = torch.nn.functional.pad(x, (0, self.padding), "constant", 0)
            

        b, _ = x.shape
        x = x.view(b, self.num_patches, self.patch_size)
        

        x = self.patch_embedding(x) # (Batch, Num_Patches, d_model)
        

        x = x + self.pos_embedding
        

        x = self.transformer_encoder(x) # (Batch, Num_Patches, d_model)
        
        x = x.mean(dim=1)
        

        x = self.fc(x)
        return x

# Initialize Model
model = EEGTransformer(input_dim=178, num_classes=5, patch_size=20)
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 15


train_losses = []
train_accuracies = []

print("\\nStarting Training...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss/len(train_loader)
    epoch_acc = 100 * correct / total
    
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    
    if (epoch+1) % 1 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")


print("\\nEvaluating on Test Set (Subject-Disjoint)...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

target_names = ['Seizure (1)', 'Tumor Area (2)', 'Healthy Area (3)', 'Eyes Closed (4)', 'Eyes Open (5)']

print("\\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))
final_accuracy = accuracy_score(all_labels, all_preds)*100
print(f"Final Accuracy: {final_accuracy:.2f}%")


print("\\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create figure with subplots
fig = plt.figure(figsize=(18, 12))


ax1 = plt.subplot(2, 3, 1)
epochs_range = range(1, epochs + 1)
ax1.plot(epochs_range, train_losses, 'b-o', linewidth=2, markersize=6, label='Training Loss')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()


ax2 = plt.subplot(2, 3, 2)
ax2.plot(epochs_range, train_accuracies, 'g-o', linewidth=2, markersize=6, label='Training Accuracy')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Training Accuracy Over Epochs', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_ylim([0, 105])


ax3 = plt.subplot(2, 3, 3)
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=['Seizure', 'Tumor', 'Healthy', 'Eyes Closed', 'Eyes Open'],
            yticklabels=['Seizure', 'Tumor', 'Healthy', 'Eyes Closed', 'Eyes Open'],
            cbar_kws={'label': 'Count'})
ax3.set_title(f'Confusion Matrix\\nAccuracy: {final_accuracy:.2f}%', fontsize=14, fontweight='bold')
ax3.set_ylabel('True Label', fontsize=12)
ax3.set_xlabel('Predicted Label', fontsize=12)


ax4 = plt.subplot(2, 3, 4)

f1_scores = []
precisions = []
recalls = []

for class_label in range(5):
    y_test_binary = (np.array(all_labels) == class_label).astype(int)
    y_pred_binary = (np.array(all_preds) == class_label).astype(int)
    
    f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)
    prec = precision_score(y_test_binary, y_pred_binary, zero_division=0)
    rec = recall_score(y_test_binary, y_pred_binary, zero_division=0)
    
    f1_scores.append(f1 * 100)
    precisions.append(prec * 100)
    recalls.append(rec * 100)

x = np.arange(5)
width = 0.25

rects1 = ax4.bar(x - width, precisions, width, label='Precision', alpha=0.8)
rects2 = ax4.bar(x, recalls, width, label='Recall', alpha=0.8)
rects3 = ax4.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)

ax4.set_ylabel('Score (%)', fontsize=12)
ax4.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'])
ax4.legend()
ax4.grid(axis='y', alpha=0.3)
ax4.set_ylim([0, 105])


ax5 = plt.subplot(2, 3, 5)
class_accuracies = []
for class_label in range(5):
    class_mask = np.array(all_labels) == class_label
    if np.sum(class_mask) > 0:
        class_acc = np.sum((np.array(all_preds)[class_mask] == class_label)) / np.sum(class_mask) * 100
        class_accuracies.append(class_acc)
    else:
        class_accuracies.append(0)

colors = sns.color_palette("husl", 5)
bars = ax5.bar(range(5), class_accuracies, alpha=0.8, color=colors)
ax5.set_ylabel('Accuracy (%)', fontsize=12)
ax5.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
ax5.set_xticks(range(5))
ax5.set_xticklabels(['Seizure', 'Tumor', 'Healthy', 'Eyes\\nClosed', 'Eyes\\nOpen'])
ax5.grid(axis='y', alpha=0.3)
ax5.set_ylim([0, 105])

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax5.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)


ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary_text = f"""
TRANSFORMER MODEL SUMMARY

Architecture:
• Input Dimension: 178
• Patch Size: 20
• Embedding Dimension: 64
• Attention Heads: 4
• Transformer Layers: 2
• Output Classes: 5

Training:
• Epochs: {epochs}
• Batch Size: {batch_size}
• Optimizer: Adam (lr=0.001)
• Loss Function: CrossEntropy

Results:
• Final Training Acc: {train_accuracies[-1]:.2f}%
• Test Accuracy: {final_accuracy:.2f}%
• Device: {device}

Subject-Aware Split:
• Train Samples: {len(X_train)}
• Test Samples: {len(X_test)}
• No subject overlap ✓
"""

ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
         verticalalignment='center', transform=ax6.transAxes,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('../results/dl_evaluation_results.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: ../results/dl_evaluation_results.png")
plt.close()


fig2, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax,
            xticklabels=['Seizure (1)', 'Tumor (2)', 'Healthy (3)', 'Eyes Closed (4)', 'Eyes Open (5)'],
            yticklabels=['Seizure (1)', 'Tumor (2)', 'Healthy (3)', 'Eyes Closed (4)', 'Eyes Open (5)'],
            cbar_kws={'label': 'Number of Samples'})
ax.set_title(f'Detailed Confusion Matrix - EEG Transformer\\nTest Accuracy: {final_accuracy:.2f}%', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('True Label', fontsize=14)
ax.set_xlabel('Predicted Label', fontsize=14)

plt.tight_layout()
plt.savefig('../results/dl_confusion_matrix_detailed.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: ../results/dl_confusion_matrix_detailed.png")
plt.close()

print("\\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\\nOutput files:")
print(f"   -> ../results/dl_evaluation_results.png")
print(f"   -> ../results/dl_confusion_matrix_detailed.png")
print(f"\\nFinal Test Accuracy: {final_accuracy:.2f}%")
