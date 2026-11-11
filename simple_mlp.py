"""Converted from simple_mlp.ipynb. Cells are separated by "# %%" markers (runnable in VS Code / PyCharm as cells)."""


# %% [markdown]
# # Lightweight MLP for Real-Time DDoS Detection in IIoT Edge Gateways
#
# This work presents a compact 4-layer multilayer perceptron (MLP) with ReLU activation and sigmoid output, trained end-to-end using binary cross-entropy loss. The model is designed for real-time DDoS detection on resource-constrained CPU-only IIoT edge gateways. To improve training stability and generalisation, the architecture incorporates dropout regularisation (0.4) and a reduced hidden layer structure (128→64→32) with a single logit output. The model is trained with a lower learning rate (0.0005) and early stopping based on validation F1 score, together with class weighting to handle natural class distributions. No downsampling or artificial balancing is applied; the model operates on the original attack ratios (CIC-DDoS2019: 77.3% attacks; Edge-IIoTset: 31.0% attacks). A calibrated threshold is selected on the validation set using the precision‑recall curve to optimise the F1 score.
#
# The detector was evaluated on two modern benchmark datasets:
# - CIC-DDoS2019 (reflection and amplification attacks)
# - Edge-IIoTset (real IIoT traffic with 15 attack families)
#
# ## Key Results
#
# - CIC-DDoS2019: Accuracy 90.82%, F1-score 0.9433, precision 0.9026, recall 98.78%, AUC 0.9597, false positive rate 36.35%, false negative rate 1.22%, average latency 0.0014 ms, 99th percentile latency 0.0109 ms, throughput 705,619 packets per second.
# - Edge-IIoTset: Accuracy 95.78%, F1-score 0.9272, precision 0.9976, recall 86.62%, AUC 0.9886, false positive rate 0.09%, false negative rate 13.38%, average latency 0.0009 ms, 99th percentile latency 0.0017 ms, throughput 1,138,305 packets per second.
#
# Both models achieve sub‑microsecond average inference latency (0.9–1.4 µs) with an extremely small memory footprint, making them suitable for deployment on typical CPU‑only IIoT edge hardware. The pipeline uses RobustScaler preprocessing and per‑dataset threshold tuning to handle realistic class distributions while keeping false positives low on the Edge‑IIoTset. However, on the more diverse CIC‑DDoS2019 dataset, the MLP exhibits a high false positive rate (36.35%), which would cause significant alert fatigue in practice. On Edge‑IIoTset, the false negative rate remains high (13.38%), meaning that more than one in ten attacks would be missed – a critical issue for safety‑critical industrial environments.
#
# The complete system is implemented in pure PyTorch and runs entirely on CPU with no GPU required. While the MLP provides a simple and extremely fast baseline, its detection performance on challenging traffic and real IIoT data is insufficient for operational deployment. This motivates the need for the more sophisticated hybrid transfer learning model presented in the companion work.


# %% [cell 2]
# CELL 1: IMPORTS AND SETUP
print("CELL 1: Loading dependencies...")
print("="*60)

import os
import gc
import random
import time
import warnings
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Settings
warnings.filterwarnings('ignore')
device = torch.device("cpu")
print(f"Using device: {device}")

# Folders
os.makedirs('saved_models_mlp', exist_ok=True)
os.makedirs('figures_mlp', exist_ok=True)

print("All dependencies loaded.")
print("="*60)


# %% [cell 3]
# CELL 2: LOAD DATASETS (NO KDDCUP99)
print("CELL 2: Loading CIC-DDoS2019 and Edge-IIoTset...")
print("="*60)

base_path = Path(r"F:\jupyter\kagglehub")  # Update if needed

paths = {
    'cic_ddos': base_path / r"datasets\dhoogla\cicddos2019\versions\3",
    'edge_iot': base_path / r"edgeiiotset-cyber-security-dataset-of-iot-iiot\versions\5\Edge-IIoTset dataset\Selected dataset for ML and DL"
}

def map_to_binary(label):
    if pd.isna(label):
        return 0
    s = str(label).lower().replace('_', '').replace(' ', '').replace('-', '')
    return 0 if any(k in s for k in ['normal', 'benign', '0']) else 1

dataset_results = {}

for name, root in paths.items():
    print(f"\nLoading {name.upper()}...")
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")
    
    files = list(root.rglob("*.csv")) + list(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No CSV/PARQUET in {root}")
    
    dfs = []
    for f in files:
        try:
            if f.suffix == '.parquet':
                df_temp = pd.read_parquet(f)
            else:
                df_temp = pd.read_csv(f, low_memory=False)
            print(f"   Loaded {f.name} → {len(df_temp):,} rows")
            dfs.append(df_temp)
            del df_temp
            gc.collect()
        except Exception as e:
            print(f"   Failed {f.name}: {e}")
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"   Total rows: {len(df):,}")
    
    # Find label column
    label_col = next((c for c in ['Label', 'label', 'Attack_type', 'Attack', 'class'] if c in df.columns), df.columns[-1])
    print(f"   Using label: '{label_col}'")
    
    df['target'] = df[label_col].apply(map_to_binary)
    print(f"   Attack ratio (natural): {df['target'].mean():.4%}")
    
    # Select numeric features only
    X_raw = df.select_dtypes(include=np.number).drop(columns=[label_col], errors='ignore')
    X = np.nan_to_num(X_raw.values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    y = df['target'].values.astype(np.int8)
    
    # Train/val/test split (70/15/15 stratified)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED)
    
    # Scale using RobustScaler (fit only on training)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    
    dataset_results[name] = {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'scaler': scaler,
        'feature_dim': X_train.shape[1]
    }
    
    del df, X, y
    gc.collect()

print("\nDatasets loaded successfully (CIC-DDoS2019 and Edge-IIoTset).")
print("No downsampling – using natural class distributions.")
print("="*60)


# %% [cell 4]
# CELL 3: MLP WITHOUT BATCHNORM (slightly weaker)
print("CELL 3: Defining MLP without BatchNorm...")
print("="*60)

class StandardMLP(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# Test
model_test = StandardMLP(dataset_results['cic_ddos']['feature_dim'])
dummy_input = torch.randn(1, dataset_results['cic_ddos']['feature_dim'])
output = model_test(dummy_input)
print(f"Model defined. Input dim: {dataset_results['cic_ddos']['feature_dim']}")
print(f"Output shape: {output.shape}")
print(f"Parameter count: {sum(p.numel() for p in model_test.parameters()):,}")
print("="*60)


# %% [cell 5]
# CELL 4: IMPROVED TRAINING FUNCTION (validation F1 + threshold calibration)
print("CELL 4: Training function with validation F1...")
print("="*60)

def train_improved_mlp(model, train_loader, val_loader, epochs, lr, patience, device, pos_weight=None):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_val_f1 = 0.0
    patience_counter = 0
    best_state_dict = None
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(Xb).squeeze()
            loss = criterion(outputs, yb.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation with threshold calibration (best F1 on validation set)
        model.eval()
        y_val_true = []
        y_val_prob = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                outputs = model(Xb).squeeze()
                probs = torch.sigmoid(outputs).cpu().numpy()
                y_val_prob.extend(probs)
                y_val_true.extend(yb.cpu().numpy())
        
        # Find best threshold on validation set
        best_thresh = 0.5
        best_f1 = 0.0
        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (np.array(y_val_prob) >= thresh).astype(int)
            f1 = f1_score(y_val_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        # Apply best threshold to get validation accuracy
        preds = (np.array(y_val_prob) >= best_thresh).astype(int)
        val_acc = accuracy_score(y_val_true, preds)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val F1: {best_f1:.4f} | Val Acc: {val_acc:.4%} | Thresh: {best_thresh:.2f}")
        
        scheduler.step(best_f1)
        
        if best_f1 > best_val_f1:
            best_val_f1 = best_f1
            best_state_dict = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(best_state_dict)
    return model, best_val_f1, best_thresh

print("Training function ready.")
print("="*60)


# %% [cell 6]
# CELL 5: TRAIN STANDARD MLP (NO BATCHNORM) ON CIC-DDoS2019 AND EDGE-IIOTSET
print("CELL 5: Training standard MLP (no BatchNorm)...")
print("="*60)

BATCH_SIZE = 256
EPOCHS = 60
PATIENCE = 10
LR = 0.0005  # lower learning rate

standard_models = {}
best_thresholds = {}

for name in ['cic_ddos', 'edge_iot']:
    print(f"\n--- Training standard MLP on {name.upper()} ---")
    data = dataset_results[name]
    
    train_dataset = TensorDataset(torch.FloatTensor(data['X_train']), torch.FloatTensor(data['y_train']))
    val_dataset = TensorDataset(torch.FloatTensor(data['X_val']), torch.FloatTensor(data['y_val']))
    test_dataset = TensorDataset(torch.FloatTensor(data['X_test']), torch.FloatTensor(data['y_test']))
    
    # drop_last=True to avoid batch size 1 issues (though no BatchNorm, still good practice)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    model = StandardMLP(data['feature_dim'], dropout_rate=0.4).to(device)
    
    # Class weighting: pos_weight = benign/attack (to give higher weight to minority)
    y_train = data['y_train']
    n_benign = np.sum(y_train == 0)
    n_attack = np.sum(y_train == 1)
    pos_weight = torch.tensor([n_benign / n_attack]).to(device) if n_attack > 0 else torch.tensor([1.0])
    print(f"Pos_weight: {pos_weight.item():.2f}")
    
    trained_model, best_f1, best_thresh = train_improved_mlp(model, train_loader, val_loader, EPOCHS, LR, PATIENCE, device, pos_weight)
    
    standard_models[name] = trained_model
    best_thresholds[name] = best_thresh
    torch.save(trained_model.state_dict(), f'saved_models_mlp/standard_mlp_{name}.pth')
    joblib.dump(data['scaler'], f'saved_models_mlp/standard_scaler_{name}.pkl')
    
    print(f"Best validation F1: {best_f1:.4f}, threshold: {best_thresh:.3f}")

print("\nStandard MLP training complete.")
print("="*60)


# %% [cell 7]
# CELL 6: EVALUATION FUNCTION WITH CALIBRATED THRESHOLD
print("CELL 6: Evaluation function...")
print("="*60)

def evaluate_improved_mlp(model, test_loader, threshold, name):
    model.eval()
    y_true = []
    y_prob = []
    inference_times = []
    
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            start = time.time()
            logits = model(Xb).squeeze().cpu().numpy()
            latency = (time.time() - start) * 1000 / len(Xb)  # ms per sample
            probs = 1 / (1 + np.exp(-logits))
            y_prob.extend(probs)
            y_true.extend(yb.numpy())
            inference_times.append(latency)
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    avg_latency = np.mean(inference_times)
    p99_latency = np.percentile(inference_times, 99)
    throughput = 1000 / avg_latency if avg_latency > 0 else 0
    
    print(f"{name} Test Results (threshold={threshold:.3f}):")
    print(f"  Accuracy: {acc:.4%} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | AUC: {auc:.4f}")
    print(f"  TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
    print(f"  FPR: {fpr:.4%} | FNR: {fnr:.4%}")
    print(f"  Avg Latency: {avg_latency:.4f} ms | 99th %: {p99_latency:.4f} ms | Throughput: {throughput:.0f} pps")
    
    return {
        'name': name,
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'auc': auc,
        'fpr': fpr,
        'fnr': fnr,
        'avg_latency_ms': avg_latency,
        'p99_latency_ms': p99_latency,
        'throughput_pps': throughput,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

print("Evaluation function ready.")
print("="*60)


# %% [cell 8]
# CELL 7: EVALUATE STANDARD MLP (NO BATCHNORM) ON TEST SETS
print("CELL 7: Evaluating standard MLP...")
print("="*60)

mlp_results = {}

for name in ['cic_ddos', 'edge_iot']:
    # Use the correct model dictionary (standard_models from CELL 5)
    model = standard_models[name]
    data = dataset_results[name]
    
    # Create test loader from stored test arrays
    X_test_tensor = torch.FloatTensor(data['X_test'])
    y_test_tensor = torch.FloatTensor(data['y_test'])
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=False)
    
    threshold = best_thresholds[name]
    res = evaluate_improved_mlp(model, test_loader, threshold, name.upper())
    mlp_results[name] = res

print("\nEvaluation complete.")
print("="*60)


# %% [cell 9]
# CELL 8: SUMMARY TABLE AND IMPROVED VISUALIZATIONS
print("CELL 8: Summary and improved visualizations...")
print("="*80)

summary = []
for name in ['cic_ddos', 'edge_iot']:
    r = mlp_results[name]
    summary.append({
        'Dataset': 'CIC-DDoS2019' if name == 'cic_ddos' else 'Edge-IIoTset',
        'Accuracy (%)': r['accuracy'] * 100,
        'F1 (%)': r['f1'] * 100,
        'Precision (%)': r['precision'] * 100,
        'Recall (%)': r['recall'] * 100,
        'Latency (us)': r['avg_latency_ms'] * 1000,
        'Throughput (pps)': r['throughput_pps']
    })

df_plot = pd.DataFrame(summary)

print("Performance Summary:")
print(df_plot.round(2).to_string(index=False))

# Table (slightly larger vertically)
fig_table, ax_table = plt.subplots(figsize=(8, 3))
ax_table.axis('off')
table = ax_table.table(cellText=df_plot.round(2).values,
                       colLabels=df_plot.columns,
                       loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)
plt.tight_layout()
plt.show()


# Figure 1
fig1, ax1 = plt.subplots(figsize=(10, 6))
x = np.arange(len(df_plot))
width = 0.2

colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

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
ax1.set_title('MLP Performance Metrics', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(df_plot['Dataset'], fontsize=11)
ax1.set_ylim(0, 105)
ax1.legend(loc='lower left', fontsize=10)
ax1.grid(True, axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('figures/mlp_scores.png', dpi=300, bbox_inches='tight')
plt.show()


# Figure 2
fig2, (ax_lat, ax_thr) = plt.subplots(1, 2, figsize=(12, 5))

# Latency 
bars_lat = ax_lat.bar(df_plot['Dataset'], df_plot['Latency (us)'],
                      color='#F68B33', edgecolor='black', linewidth=0.5)

lat_range = df_plot['Latency (us)'].max() - df_plot['Latency (us)'].min()
lat_offset = lat_range * 0.08 if lat_range > 0 else 1  # slightly more space

for bar, val in zip(bars_lat, df_plot['Latency (us)']):
    ax_lat.text(bar.get_x() + bar.get_width()/2., val + lat_offset,
                f'{val:.1f} µs',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax_lat.set_ylabel('Latency (microseconds)', fontsize=12, fontweight='bold')
ax_lat.set_title('Average Inference Latency', fontsize=13, fontweight='bold', pad=20)

ax_lat.set_ylim(0, df_plot['Latency (us)'].max() * 1.25)

ax_lat.grid(True, axis='y', alpha=0.3, linestyle='--')


# ---- Throughput ----
bars_thr = ax_thr.bar(df_plot['Dataset'], df_plot['Throughput (pps)'],
                      color='#2E8B57', edgecolor='black', linewidth=0.5)

thr_range = df_plot['Throughput (pps)'].max() - df_plot['Throughput (pps)'].min()
thr_offset = thr_range * 0.08 if thr_range > 0 else 50  # more spacing

for bar, val in zip(bars_thr, df_plot['Throughput (pps)']):
    ax_thr.text(bar.get_x() + bar.get_width()/2., val + thr_offset,
                f'{val:.0f} pps',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax_thr.set_ylabel('Throughput (packets/sec)', fontsize=12, fontweight='bold')
ax_thr.set_title('Throughput', fontsize=13, fontweight='bold')

#  give free space above bars
ax_thr.set_ylim(0, df_plot['Throughput (pps)'].max() * 1.25)

ax_thr.grid(True, axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('figures/mlp_realtime.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualizations saved to figures_mlp/")
print("  → mlp_scores.png")
print("  → mlp_realtime.png")
print("="*80)


# %% [cell 10]
# CELL 9: CONFUSION MATRICES
print("CELL 9: Confusion matrices...")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, name in enumerate(['cic_ddos', 'edge_iot']):
    r = mlp_results[name]
    cm = confusion_matrix(r['y_true'], r['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
    axes[idx].set_title(f' MLP - Confusion Matrix {name.upper()}')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('figures/improved_mlp_confusion.png', dpi=300)
plt.show()
print("Confusion matrices saved.")
print("="*60)


# %% [cell 11]
# CELL 8B: SEPARATE CONFUSION MATRICES MLP Baseline
print("CELL 9: Separate confusion matrices for MLP Baseline...")
print("="*60)

for name in ['cic_ddos', 'edge_iot']:
    r = mlp_results[name]
    cm = confusion_matrix(r['y_true'], r['y_pred'])
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'],
                cbar=False, annot_kws={"size": 14, "weight": "bold"})
    plt.title(f'MLP Baseline - Confusion Matrix {name.upper()}', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted', fontsize=10)
    plt.ylabel('Actual', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'figures/mlp_cm_{name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved confusion matrix for {name.upper()} to figures/mlp_cm_{name}.png")

print("="*60)


# %% [cell 12]
# CELL 10: MLP RESULTS TABLE
print("CELL 11: MLP final results table (with AUC)...")
print("="*80)

data = []
for name in ['cic_ddos', 'edge_iot']:
    r = mlp_results[name]
    data.append({
        'Dataset': 'CIC-DDoS2019' if name == 'cic_ddos' else 'Edge-IIoTset',
        'Accuracy (%)': round(r['accuracy'] * 100, 2),
        'F1': round(r['f1'], 4),
        'Precision': round(r['precision'], 4),
        'Recall (%)': round(r['recall'] * 100, 2),
        'AUC': round(r['auc'], 4),
        'FPR (%)': round(r['fpr'] * 100, 2),
        'FNR (%)': round(r['fnr'] * 100, 2),
        'Latency (ms)': round(r['avg_latency_ms'], 4),
        'Throughput (pps)': int(r['throughput_pps'])
    })

df_mlp = pd.DataFrame(data)
print("\nMLP Performance Summary (including AUC):")
print(df_mlp.to_string(index=False))

df_mlp.to_csv('mlp_final_results.csv', index=False)
print("\nResults saved to mlp_final_results.csv")
print("="*80)
