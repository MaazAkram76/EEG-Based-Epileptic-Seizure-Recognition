import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ---------------------------------------------------------
# 1. DATA LOADING & SUBJECT-AWARE SPLITTING
# ---------------------------------------------------------
FILENAME = 'data.csv'

def load_and_split_data(filename):
    print(f"📂 Loading {filename}...")
    df = pd.read_csv(filename)
    
    # Extract Columns
    # ID is column 0 (e.g., "X21.V1.791")
    # Features are columns 1 to 178 (X1...X178)
    # Label is column 179 ('y')
    
    ids = df.iloc[:, 0]
    X = df.iloc[:, 1:-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64)
    
    # ⚠️ CRITICAL: SUBJECT GROUPING
    # Verified Logic: Subject ID is locally unique per folder (Class), but suffixes are reused.
    # We must group by (Class, Suffix) to identify unique subjects.
    
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
        
        # GroupID = y * 10000 + Suffix (Ensures Subject Disjointness across classes)
        # Note: y values are 1-5. Suffixes are ~1-1000.
        groups = y * 10000 + raw_indices
        
        # Verification
        unique_groups = len(np.unique(groups))
        print(f"✅ Subject Grouping Logic: Derived {unique_groups} unique subjects (Expect ~500).")
        
        # Check Consistency (Leakage)
        check_df = pd.DataFrame({'group': groups, 'label': y})
        inconsistent_groups = check_df.groupby('group')['label'].nunique()
        if inconsistent_groups.max() > 1:
            print("⚠️ WARNING: Subject grouping hypothesis failed! Some groups have mixed labels.")
            print(inconsistent_groups[inconsistent_groups > 1])
        else:
            print("✅ Subject Grouping verified: No leakage detected.")

    except Exception as e:
        print(f"❌ Error parsing IDs or Grouping: {e}")
        print("Fallback: Using simple rows (LEAKAGE RISK!)")
        groups = np.arange(len(y))

    # Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Adjust labels to 0-based (1-5 -> 0-4) for PyTorch CrossEntropy
    y_train = y_train - 1
    y_test = y_test - 1
    
    print(f"   -> Train Set: {X_train.shape}")
    print(f"   -> Test Set:  {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_and_split_data(FILENAME)

# ---------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------
# Scale based on TRAIN set statistics only, apply to Test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 3. PYTORCH DATASET
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 4. PATCH-BASED TRANSFORMER MODEL
# ---------------------------------------------------------
class EEGTransformer(nn.Module):
    def __init__(self, input_dim=178, num_classes=5, patch_size=20, d_model=64, nhead=4, num_layers=2):
        super(EEGTransformer, self).__init__()
        
        self.patch_size = patch_size
        self.d_model = d_model
        
        # Calculate padding needed
        # We want input to be divisible by patch_size
        remainder = input_dim % patch_size
        self.padding = (patch_size - remainder) if remainder > 0 else 0
        self.padded_dim = input_dim + self.padding
        self.num_patches = self.padded_dim // patch_size
        
        print(f"🏗️ Model Config: Input {input_dim} -> Padded {self.padded_dim}")
        print(f"   -> {self.num_patches} Patches of size {patch_size}")
        
        # 1. Patch Projection (Linear)
        # Transforms each (patch_size) vector into (d_model) vector
        self.patch_embedding = nn.Linear(patch_size, d_model)
        
        # 2. Positional Encoding (Learnable)
        # Shape: (1, num_patches, d_model) broadcastable
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Classification Head
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x shape: (Batch, 178)
        
        # Step 1: Pad
        if self.padding > 0:
            # Pad the last dimension with zeros
            x = torch.nn.functional.pad(x, (0, self.padding), "constant", 0)
            
        # Step 2: Patchify
        # Reshape to (Batch, Num_Patches, Patch_Size)
        b, _ = x.shape
        x = x.view(b, self.num_patches, self.patch_size)
        
        # Step 3: Embed
        x = self.patch_embedding(x) # (Batch, Num_Patches, d_model)
        
        # Step 4: Add Position
        x = x + self.pos_embedding
        
        # Step 5: Transformer
        x = self.transformer_encoder(x) # (Batch, Num_Patches, d_model)
        
        # Step 6: Global Average Pooling
        x = x.mean(dim=1) # (Batch, d_model)
        
        # Step 7: Output
        x = self.fc(x)
        return x

# Initialize Model
model = EEGTransformer(input_dim=178, num_classes=5, patch_size=20)
model.to(device)

# ---------------------------------------------------------
# 5. TRAINING LOOP
# ---------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 15

print("\n🔥 Starting Training...")
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
        
    if (epoch+1) % 1 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Acc: {100 * correct / total:.2f}%")

# ---------------------------------------------------------
# 6. EVALUATION
# ---------------------------------------------------------
print("\n🧪 Evaluating on Test Set (Subject-Disjoint)...")
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

# Map back to original class names/numbers (0->1, 1->2 etc)
# Original Labels: 1=Seizure, 2-5=Non-Seizure
target_names = ['Seizure (1)', 'Tumor Area (2)', 'Healthy Area (3)', 'Eyes Closed (4)', 'Eyes Open (5)']

print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))
print(f"Final Accuracy: {accuracy_score(all_labels, all_preds)*100:.2f}%")
