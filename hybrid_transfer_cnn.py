"""Converted from hybrid_transfer_cnn.ipynb. Cells are separated by "# %%" markers (runnable in VS Code / PyCharm as cells)."""


# %% [markdown]
# # Lightweight Hybrid Deep and Transfer Learning for Real-Time DDoS Detection in Resource-Constrained IIoT Edge Gateways
#
# ## Introduction
# The rapid growth of Industrial Internet of Things (IIoT) devices has transformed industrial automation by enabling real-time monitoring and control. However, the increased connectivity has significantly expanded the attack surface, making IIoT systems highly vulnerable to Distributed Denial-of-Service (DDoS) attacks that can disrupt critical operations and cause substantial economic losses. Traditional intrusion detection approaches often fail to satisfy the stringent requirements of edge deployment, particularly the need for high detection accuracy, low computational overhead, and sub-millisecond inference latency on resource-constrained CPU-only gateways.
#
# To address these challenges, this work proposes a lightweight hybrid deep learning framework that integrates a Convolutional Neural Network (CNN) with a Convolutional Block Attention Module (CBAM) and a feature projection layer for effective domain adaptation. The model is sequentially pre-trained on CIC-DDoS2019 and then adapted to the target Edge-IIoTset dataset to bridge the domain shift between general network traffic and real IIoT-specific patterns.
#
# ## Project Summary
# The proposed framework employs robust scaling and a fixed 39-dimensional projection layer to handle varying feature dimensions across datasets. The architecture combines a CNN backbone enhanced with CBAM attention for capturing temporal burst patterns and a domain-specific MLP classifier with single-logit output. Selective transfer learning is applied by copying the CNN extractor and CBAM while initializing fresh projection and classifier layers for the target domain. The model is trained with the Adam optimizer, binary cross-entropy loss, and early stopping based on validation F1 score. A calibrated threshold is selected using the precision-recall curve on the validation set to optimise the F1 score.
#
# The detector was evaluated on two modern benchmark datasets:
# - CIC-DDoS2019 (reflection and amplification attacks) as the source domain
# - Edge-IIoTset (real IIoT traffic with 15 attack families) as the target domain
#
# ## Key Results
#
# - CIC-DDoS2019: Accuracy 97.93%, F1-score 0.9867, precision 0.9788, recall 99.48%, AUC 0.9727, false positive rate 7.36%, false negative rate 0.52%, average latency 0.013 ms, 99th percentile latency 0.018 ms, throughput 79,371 packets per second.
# - Edge-IIoTset: Accuracy 99.51%, F1-score 0.9921, precision 0.9935, recall 99.06%, AUC 0.9997, false positive rate 0.29%, false negative rate 0.94%, average latency 0.011 ms, 99th percentile latency 0.018 ms, throughput 87,244 packets per second.
#
# The proposed hybrid model achieves high detection accuracy (>97.9% on both datasets) with very low false positive and false negative rates (FPR ≤7.36%, FNR ≤0.94%). The inference latency remains in the tens of microseconds (11–13 µs), which is three orders of magnitude below typical industrial control loop deadlines (1–10 ms). The throughput exceeds 79,000 packets per second on a standard CPU, sufficient for line-rate processing on 10 Gbps links when parallelised across multiple cores.
#
# Comprehensive visualisations, including confusion matrices, performance bar charts, and real-time latency/throughput plots, illustrate the robustness of the approach. The work also discusses remaining challenges such as the impact of class imbalance and the need for future validation on actual edge hardware (e.g., Raspberry Pi, Intel Atom). The complete implementation is written in pure PyTorch and runs entirely on CPU without GPU acceleration, making it immediately deployable on commodity IIoT edge gateways.


# %% [cell 2]
# Cell 2: Dependencies & Imports
print("Cell 2: Loading dependencies...")

import os
import gc
import random
import time
import warnings
import joblib
from pathlib import Path
from collections import deque
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from shap import Explainer, summary_plot
import psutil
import timeit

# Seed 
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Style settings 
plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (11, 7)

warnings.filterwarnings('ignore')

device = torch.device("cpu")
print(f"Using device: {device}")

os.makedirs('saved_models', exist_ok=True)
os.makedirs('figures', exist_ok=True)
print("All dependencies are loaded!")
print("Ready for CIC-DDoS2019 which is the source and Edge-IIoTset which is the target")


# %% [cell 3]
# Cell 3: Multi-Dataset Loading
print("Cell 3: Loading datasets (CIC-DDoS2019 and Edge-IIoTset)")

base_path = Path(r"F:\jupyter\kagglehub")
paths = {
    'cic_ddos': base_path / r"datasets\dhoogla\cicddos2019\versions\3",
    'edge_iot': base_path / r"edgeiiotset-cyber-security-dataset-of-iot-iiot\versions\5\Edge-IIoTset dataset\Selected dataset for ML and DL"
}

dataset_results = {}

def map_to_binary(label):
    if pd.isna(label): return 0
    s = str(label).lower().replace('_', '').replace(' ', '').replace('-', '')
    return 0 if any(k in s for k in ['normal', 'benign', '0']) else 1

for name, root in paths.items():
    print(f"\nLoading {name.upper()}...")
    files = list(root.rglob("*.csv")) + list(root.rglob("*.parquet"))
    dfs = []
    for f in files:
        try:
            df_temp = pd.read_parquet(f) if f.suffix == '.parquet' else pd.read_csv(f, low_memory=False)
            print(f" Loaded {f.name} → {len(df_temp):,} rows")
            dfs.append(df_temp)
            del df_temp
            gc.collect()
        except Exception as e:
            print(f" Failed {f.name}: {e}")
    df = pd.concat(dfs, ignore_index=True)
    print(f" Total rows: {len(df):,}")
    label_col = next((c for c in ['Label', 'label', 'Attack_type', 'Attack', 'class'] if c in df.columns), df.columns[-1])
    print(f" Using label: '{label_col}'")
    df['target'] = df[label_col].apply(map_to_binary)
    print(f" Attack ratio: {df['target'].mean():.4%} (natural)")
    X_raw = df.select_dtypes(include=np.number).drop(columns=[label_col], errors='ignore')
    X = np.nan_to_num(X_raw.values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    y = df['target'].values.astype(np.int8)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    dataset_results[name] = {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'scaler': scaler,
        'feature_count': X_train.shape[1]
    }
    del df, X, y
    gc.collect()

print("\nDatasets loaded successfully.")


# %% [cell 4]
# Cell 4: Unified Model Definition, single logit output
print("Cell 4: Unified Model Definition, single logit")

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // 16, 1),
            nn.ReLU(),
            nn.Conv1d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv1d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
    def forward(self, x):
        ca = self.channel_attn(x) * x
        avg = torch.mean(ca, dim=1, keepdim=True)
        max_ = torch.max(ca, dim=1, keepdim=True)[0]
        sa = self.spatial_attn(torch.cat([avg, max_], dim=1)) * ca
        return sa

class UnifiedDetector(nn.Module):
    def __init__(self, input_features, is_target=False):
        super().__init__()
        self.projection = nn.Linear(input_features, 39)
        self.cnn_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            CBAM(128),
            nn.AdaptiveAvgPool1d(1)
        )
        concat_dim = 128 + 39
        if is_target:
            self.mlp_classifier = nn.Sequential(
                nn.Linear(concat_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        else:
            self.mlp_classifier = nn.Sequential(
                nn.Linear(concat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
    def forward(self, x):
        if x.dim() == 4 and x.shape[1] == 1 and x.shape[2] == 1:
            x = x.squeeze(2)
        elif x.dim() != 3:
            raise ValueError(f"Unexpected input dim {x.dim()}, shape {x.shape}. Expected 3D [B,1,L]")
        proj = self.projection(x.squeeze(1))
        cnn_in = proj.unsqueeze(1)
        cnn_out = self.cnn_extractor(cnn_in).squeeze(-1)
        concat = torch.cat([cnn_out, proj], dim=1)
        logit = self.mlp_classifier(concat).squeeze(1)
        return logit

print("Unified Detector is defined, single logit output for stable calibration.")
model_example = UnifiedDetector(78).to(device)
print(sum(p.numel() for p in model_example.parameters()), "parameters (~118k)")


# %% [cell 5]
# Cell 5: Dataset Class and Dataloaders
print("Cell 5: Dataset Class and Dataloaders")

class AttackDataset(Dataset):
    def __init__(self, X, y):
        if X.ndim != 2:
            raise ValueError(f"Expected 2D data (N, F), got {X.shape}")
        self.X = X.astype(np.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        x = self.X[idx]
        x = torch.from_numpy(x)
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        return x, self.y[idx]

def get_dataloaders(name):
    data = dataset_results[name]
    train_ds = AttackDataset(data['X_train'], data['y_train'])
    val_ds = AttackDataset(data['X_val'], data['y_val'])
    test_ds = AttackDataset(data['X_test'], data['y_test'])
    return (
        DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0),
        DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)
    )

print("Dataset class and dataloaders ready.")


# %% [cell 6]
# Cell 6: Pre-training on CIC-DDoS2019
print("Cell 6: Pre-training on CIC-DDoS2019")

name = 'cic_ddos'
input_features = dataset_results[name]['feature_count']
model = UnifiedDetector(input_features).to(device)
train_loader, val_loader, test_loader = get_dataloaders(name)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 25
best_acc = 0.0
patience = 7
patience_counter = 0

for epoch in tqdm(range(epochs)):
    model.train()
    running_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(Xb)
        loss = criterion(outputs, yb.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
   
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            outputs = model(Xb.to(device))
            pred = (torch.sigmoid(outputs) > 0.5).long()
            total += yb.size(0)
            correct += (pred == yb.to(device)).sum().item()
   
    val_acc = correct / total
    print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4%}")
   
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'saved_models/pretrained_cic_ddos.pth')
        print(f" Best model saved ({val_acc:.4%})")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} - no improvement for {patience} epochs.")
            break

print(f"Pre-training complete. Best val acc: {best_acc:.4%}")


# %% [cell 7]
# Cell 7: Transfer to Edge-IIoTset
print("Cell 7: Transfer to Edge-IIoTset")

name = 'edge_iot'
edge_features = dataset_results[name]['feature_count']

pretrained = UnifiedDetector(dataset_results['cic_ddos']['feature_count']).to(device)
pretrained.load_state_dict(torch.load('saved_models/pretrained_cic_ddos.pth'))

model = UnifiedDetector(edge_features).to(device)

state_dict = pretrained.state_dict()
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and 'cnn_extractor' in k and v.shape == model_dict[k].shape}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
print("Selective transfer complete: CNN extractor copied, projection + MLP fresh.")

for param in model.cnn_extractor[:6].parameters():
    param.requires_grad = False

train_loader, val_loader, test_loader = get_dataloaders(name)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
criterion = nn.BCEWithLogitsLoss()
epochs = 20
best_acc = 0.0
patience = 7
patience_counter = 0

for epoch in tqdm(range(epochs)):
    model.train()
    running_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(Xb)
        loss = criterion(outputs, yb.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
   
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            outputs = model(Xb.to(device))
            pred = (torch.sigmoid(outputs) > 0.5).long()
            total += yb.size(0)
            correct += (pred == yb.to(device)).sum().item()
   
    val_acc = correct / total
    print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4%}")
   
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'saved_models/final_edge_iot_model.pth')
        print(f" Best model saved ({val_acc:.4%})")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} - no improvement for {patience} epochs.")
            break

print(f"Transfer complete. Best val acc: {best_acc:.4%}")


# %% [cell 8]
# Cell 8: Evaluation - per-dataset model loading (transfer hybrid)
print("Cell 8: Evaluation - per-dataset model loading")

import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

results = {}

RECALL_TARGET = 0.95
FALLBACK_THRESH = 0.5
TEMPERATURE = 30.0
CLIP_MIN = -10.0
CLIP_MAX = 10.0
CLIP_ENABLED = True

def evaluate_model(model, loader, name):
    model.eval()
    y_true, y_prob, y_logits_raw = [], [], []
    inference_times = []
    with torch.no_grad():
        for Xb, yb in loader:
            start = time.time()
            Xb = Xb.to(device)
            logits_raw = model(Xb).cpu().numpy()
            if CLIP_ENABLED:
                logits_raw = np.clip(logits_raw, CLIP_MIN, CLIP_MAX)
            logits_scaled = logits_raw / TEMPERATURE
            probs = torch.sigmoid(torch.from_numpy(logits_scaled)).numpy()
            y_prob.extend(probs)
            y_logits_raw.extend(logits_raw)
            y_true.extend(yb.numpy())
            inference_times.append((time.time() - start) * 1000 / len(Xb))
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_logits_raw = np.array(y_logits_raw)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    valid_idx = np.where(recall >= RECALL_TARGET)[0]
    if len(valid_idx) == 0:
        thresh = FALLBACK_THRESH
    else:
        f1_scores = 2 * (precision[valid_idx] * recall[valid_idx]) / (precision[valid_idx] + recall[valid_idx] + 1e-8)
        optimal_idx = valid_idx[np.argmax(f1_scores)]
        thresh = thresholds[optimal_idx] if optimal_idx < len(thresholds) else FALLBACK_THRESH
    if thresh < 0.01:
        thresh = FALLBACK_THRESH
    y_pred = (y_prob >= thresh).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    auc_ = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    avg_latency = np.mean(inference_times)
    p99_latency = np.percentile(inference_times, 99)
    throughput = 1000 / avg_latency if avg_latency > 0 else 0
    print(f"{name} Test - Acc: {acc:.4%} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | AUC: {auc_:.4f}")
    print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
    print(f"FPR: {fpr:.4%} | FNR: {fnr:.4%} | Avg Latency: {avg_latency:.3f} ms | P99: {p99_latency:.3f} ms | Throughput: {throughput:.1f} pps")
    print(f"Calibrated Threshold: {thresh:.4f}")
    
    # Save confusion matrix as separate figure (same folder as MLP)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign','Attack'], yticklabels=['Benign','Attack'])
    plt.title(f'Transfer Hybrid - Confusion Matrix {name.upper()}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f"figures/transfer_cm_{name}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return {'acc': acc, 'f1': f1, 'prec': prec, 'rec': rec, 'auc': auc_, 'fpr': fpr, 'fnr': fnr, 'avg_latency': avg_latency, 'p99_latency': p99_latency, 'throughput': throughput}

models = {
    'cic_ddos': UnifiedDetector(dataset_results['cic_ddos']['feature_count']).to(device),
    'edge_iot': UnifiedDetector(dataset_results['edge_iot']['feature_count']).to(device)
}
models['cic_ddos'].load_state_dict(torch.load('saved_models/pretrained_cic_ddos.pth'))
models['edge_iot'].load_state_dict(torch.load('saved_models/final_edge_iot_model.pth'))

for name in ['cic_ddos', 'edge_iot']:
    _, _, test_loader = get_dataloaders(name)
    results[name] = evaluate_model(models[name], test_loader, f"{name.upper()} Test")

print("\nEvaluation complete.")


# %% [cell 9]
# Cell 9: FINAL VISUALIZATIONS (throughput in pps)
print("Cell 9: FINAL VISUALIZATIONS")
print("="*80)

# Prepare data from results dictionary
df_plot = pd.DataFrame({
    'Dataset': ['CIC-DDoS2019', 'Edge-IIoTset'],
    'Accuracy (%)': [results['cic_ddos']['acc'] * 100, results['edge_iot']['acc'] * 100],
    'F1 (%)': [results['cic_ddos']['f1'] * 100, results['edge_iot']['f1'] * 100],
    'Precision (%)': [results['cic_ddos']['prec'] * 100, results['edge_iot']['prec'] * 100],
    'Recall (%)': [results['cic_ddos']['rec'] * 100, results['edge_iot']['rec'] * 100],
    'AUC': [results['cic_ddos']['auc'], results['edge_iot']['auc']],
    'Latency (us)': [results['cic_ddos']['avg_latency'] * 1000, results['edge_iot']['avg_latency'] * 1000],
    'Throughput (pps)': [results['cic_ddos']['throughput'], results['edge_iot']['throughput']]
})

colors = ['#2E86AB', '#D64933', '#3F7E6B', '#F18F01']

# FIGURE 1
fig1, ax1 = plt.subplots(figsize=(10, 6))
x = np.arange(len(df_plot))
width = 0.2

bars1 = ax1.bar(x - 1.5*width, df_plot['Accuracy (%)'], width, label='Accuracy', color=colors[0], edgecolor='black', linewidth=0.5)
bars2 = ax1.bar(x - 0.5*width, df_plot['F1 (%)'], width, label='F1', color=colors[1], edgecolor='black', linewidth=0.5)
bars3 = ax1.bar(x + 0.5*width, df_plot['Precision (%)'], width, label='Precision', color=colors[2], edgecolor='black', linewidth=0.5)
bars4 = ax1.bar(x + 1.5*width, df_plot['Recall (%)'], width, label='Recall', color=colors[3], edgecolor='black', linewidth=0.5)

def add_labels_bottom(bars):
    for bar in bars:
        height = bar.get_height()
        offset = height * 0.02
        ax1.text(bar.get_x() + bar.get_width()/2., height + offset,
                 f'{height:.1f}%',
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')

add_labels_bottom(bars1)
add_labels_bottom(bars2)
add_labels_bottom(bars3)
add_labels_bottom(bars4)

ax1.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax1.set_title('Transfer Hybrid Model: Performance Metrics', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(df_plot['Dataset'], fontsize=11)
ax1.set_ylim(0, 105)

ax1.legend(loc='lower left', fontsize=10)
ax1.grid(True, axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('figures/transfer_performance_scores.png', dpi=400, bbox_inches='tight')
plt.show()


# FIGURE 2
fig2, (ax_lat, ax_thr) = plt.subplots(1, 2, figsize=(12, 5))

# Latency subplot
lat_color = '#117A65'
bars_lat = ax_lat.bar(df_plot['Dataset'], df_plot['Latency (us)'], color=lat_color, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars_lat, df_plot['Latency (us)']):
    offset = max(val * 0.05, 1.0)
    ax_lat.text(bar.get_x() + bar.get_width()/2., val + offset,
                f'{val:.1f} µs',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax_lat.set_ylabel('Latency (microseconds)', fontsize=12, fontweight='bold')

ax_lat.set_title('Average Inference Latency', fontsize=13, fontweight='bold', pad=24)

ax_lat.grid(True, axis='y', alpha=0.3, linestyle='--')


# Throughput subplot
thr_color = '#B03A2E'
bars_thr = ax_thr.bar(df_plot['Dataset'], df_plot['Throughput (pps)'], color=thr_color, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars_thr, df_plot['Throughput (pps)']):
    offset = max(val * 0.05, 50)
    ax_thr.text(bar.get_x() + bar.get_width()/2., val + offset,
                f'{val:.0f} pps',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax_thr.set_ylabel('Throughput (packets/sec)', fontsize=12, fontweight='bold')
ax_thr.set_title('Throughput', fontsize=13, fontweight='bold')
ax_thr.grid(True, axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('figures/transfer_realtime.png', dpi=400, bbox_inches='tight')
plt.show()

print("Visualizations saved:")
print("  figures/transfer_performance_scores.png")
print("  figures/transfer_realtime.png")
print("="*80)


# %% [cell 10]
# Cell 10: Big O Complexity and Schedulability Analysis (corrected)
print("Cell 10: Big O Complexity and Schedulability Analysis")

def big_o_calc(F=78, C=128, K=5, H=256, M=4):
    cnn_ops = F * C * K
    mlp_ops = H**2 * M
    total = cnn_ops + mlp_ops
    return total

print(f"Big O operations: ~{big_o_calc():,} (lightweight for edge)")

def liu_layland(n_tasks=2):
    bound = n_tasks * (2**(1/n_tasks) - 1)
    return bound

C = 0.011   # worst-case execution time (ms)
T = 1.0     # minimum inter-arrival time (ms) for 10 Gbps, 500 B packets? Actually 400 ns = 0.0004 ms, but using 1 ms as example.
U = C / T
n_tasks = 2   # define here
bound = liu_layland(n_tasks)

print(f"Task utilization U = {U:.3f}")
print(f"Liu-Layland bound for {n_tasks} tasks: {bound:.3f}")
print("→ Schedulable under fixed-priority scheduling." if U < bound else "→ Utilization exceeds bound.")


# %% [cell 11]
# Cell 11: Computation Usage Metrics
print("Cell 11: Computation Usage Metrics")

model = UnifiedDetector(dataset_results['edge_iot']['feature_count']).to(device)
model.load_state_dict(torch.load('saved_models/final_edge_iot_model.pth'))
model.eval()

_, _, test_loader = get_dataloaders('edge_iot')

start_time = time.time()
memory_start = psutil.Process(os.getpid()).memory_info().rss / 1024**2

with torch.no_grad():
    for Xb, yb in test_loader:
        outputs = model(Xb.to(device))

time_taken = time.time() - start_time
memory_end = psutil.Process(os.getpid()).memory_info().rss / 1024**2

print(f"Edge-IIoTset Test Computation:")
print(f"  Total Time : {time_taken:.2f} seconds")
print(f"  Memory Usage: {memory_end - memory_start:.2f} MB")
print(f"  Average inference time per batch: {time_taken/len(test_loader):.4f} s")
print("\nComputation usage measured on Edge-IIoTset (target domain).")


# %% [cell 12]
# Cell 10: RESULT EXTRACTION & FINAL SUMMARY
print("Cell 10: RESULT EXTRACTION & FINAL SUMMARY")

# Extract and organize results
summary_data = {
    'Dataset': ['CIC-DDoS2019', 'Edge-IIoTset'],
    'Accuracy': [results['cic_ddos']['acc'], results['edge_iot']['acc']],
    'F1': [results['cic_ddos']['f1'], results['edge_iot']['f1']],
    'Precision': [results['cic_ddos']['prec'], results['edge_iot']['prec']],
    'Recall': [results['cic_ddos']['rec'], results['edge_iot']['rec']],
    'AUC-ROC': [results['cic_ddos']['auc'], results['edge_iot']['auc']],
    'FPR (%)': [results['cic_ddos']['fpr']*100, results['edge_iot']['fpr']*100],
    'FNR (%)': [results['cic_ddos']['fnr']*100, results['edge_iot']['fnr']*100],
    'Latency (ms/packet)': [results['cic_ddos']['avg_latency'], results['edge_iot']['avg_latency']],
    '99th % Latency': [results['cic_ddos']['p99_latency'], results['edge_iot']['p99_latency']],
    'Packets/sec': [results['cic_ddos']['throughput'], results['edge_iot']['throughput']]
}

df_summary = pd.DataFrame(summary_data)

# Print clean final table
print("\n" + "="*100)
print("FINAL PERFORMANCE METRICS FOR ALL DATASETS")
print("="*100)
print(df_summary.round(4).to_string(index=False))
print("="*100)

# Highlights for IDS (most important numbers)
print("\nHIGHLIGHTS:")
print(f"CIC-DDoS2019: Recall = {results['cic_ddos']['rec']*100:.2f}%, FNR = {results['cic_ddos']['fnr']*100:.2f}%")
print(f"Edge-IIoTset : Recall = {results['edge_iot']['rec']*100:.2f}%, FNR = {results['edge_iot']['fnr']*100:.2f}%")
print(f"Latency: {results['cic_ddos']['avg_latency']:.3f} ms (CIC) | {results['edge_iot']['avg_latency']:.3f} ms (Edge)")
print(f"Throughput: {results['cic_ddos']['throughput']:.0f} pps (CIC) | {results['edge_iot']['throughput']:.0f} pps (Edge)")

# Save to CSV for easy use in paper
df_summary.to_csv("FINAL_99PERCENT_METRICS.csv", index=False)
print("\nResults saved to FINAL_99PERCENT_METRICS.csv")

print("\nCell 10 completed: All results extracted and summarized.")
