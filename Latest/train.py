
import os
import sys
import time
import copy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
)

try:
    from thop import profile as thop_profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

# =============================================================================
# Paths
# =============================================================================

sys.path.append("/ssd_scratch/abnp/ecg_on_edge/2/ltafdb")

LTAFDB_DIR  = "/ssd_scratch/abnp/ecg_on_edge/data/binary_class/windowed"
RESULTS_DIR = "/ssd_scratch/abnp/ecg_on_edge/2/ltafdb/results_v3"
MODEL_PATH  = "/ssd_scratch/abnp/ecg_on_edge/2/ltafdb/tinyresecg_v3.pth"

os.makedirs(RESULTS_DIR, exist_ok=True)

from ecg_dataset import ECGDataset
from model import TinyResECG

# =============================================================================
# Reproducibility
# =============================================================================

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# =============================================================================
# Hyperparameters
# =============================================================================

BATCH_SIZE   = 32
EPOCHS       = 20
LR           = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 6       # increased — don't stop at epoch 3 again
DROPOUT      = 0.35
WARMUP_EPOCHS = 3       # linear LR warmup before cosine decay
NUM_WORKERS  = 2

# Class weights — AFIB upweighted because recall=91.6% is the weak link
# Normal recall is already 99.8% so we can safely reduce its weight
# Normal=0.6, AFIB=1.4 gives a gentle but meaningful push toward AFIB recall
CLASS_WEIGHTS = [0.6, 1.4]

# Focal loss settings
FOCAL_GAMMA = 2.0   # focuses training on the hard AFIB->Normal misclassifications

# Mixup
USE_MIXUP   = True
MIXUP_ALPHA = 0.2

NUM_CLASSES = 2
LABELS      = ["Normal", "AFIB"]

# =============================================================================
# Patient splits
# =============================================================================

TRAIN_PATIENTS = [
    "55","39","117","33","35","111","07","56","115","16","120",
    "00","01","03","05","06","08","10","100","101","102","103",
    "104","105","110","112","113","114","116","119","121","122",
    "13","15","17","18","19","20","21","22","23","24","25","26",
    "28","30","32","34","37","38","60","64","65","68","69","70","71","75",
]
VALID_PATIENTS = ["45","51","44","49","42","43","47","48"]
TEST_PATIENTS  = ["118","11","72","74","62","53","54","58"]

# =============================================================================
# Focal Loss
# =============================================================================

class FocalLoss(nn.Module):
    

    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight   # (num_classes,) tensor

    def forward(self, inputs, targets):
        # Standard CE (no reduction) — respects class weights
        ce = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")

        # Probability of the true class
        probs = F.softmax(inputs, dim=1)
        pt    = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal modulation: down-weight easy examples
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()

# =============================================================================
# LR warmup + cosine decay scheduler
# =============================================================================

def build_scheduler(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    """
    Linear warmup for warmup_epochs, then cosine decay to min_lr.
    Avoids the large LR at epoch 1 that caused best_epoch=3 last time.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs          # linear ramp 0→1
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cosine   = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(cosine, min_lr / LR)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# =============================================================================
# Mixup
# =============================================================================

def mixup_batch(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

# =============================================================================
# Dataset stats printer
# =============================================================================

def print_split_stats(name, dataset):
    counts = np.zeros(NUM_CLASSES, dtype=int)
    for patient, idx in dataset.index:
        _, y = dataset.data_cache[patient]
        counts[int(y[idx])] += 1
    total = counts.sum()
    print(f"\n  ===== {name} =====")
    print(f"  Total : {total:,}")
    for i, lbl in enumerate(LABELS):
        print(f"  {lbl:8s}: {counts[i]:7,}  ({100*counts[i]/total:.1f}%)")

# =============================================================================
# Device & data
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

print("\n===== Loading datasets =====")
train_dataset = ECGDataset(TRAIN_PATIENTS, LTAFDB_DIR, is_train=True)
valid_dataset = ECGDataset(VALID_PATIENTS, LTAFDB_DIR, is_train=False)
test_dataset  = ECGDataset(TEST_PATIENTS,  LTAFDB_DIR, is_train=False)

print_split_stats("TRAIN", train_dataset)
print_split_stats("VALID", valid_dataset)
print_split_stats("TEST",  test_dataset)

# No WeightedRandomSampler — natural distribution, loss weights handle imbalance
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

# =============================================================================
# Model
# =============================================================================

model = TinyResECG(num_classes=NUM_CLASSES, dropout=DROPOUT).to(device)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
param_size_mb    = total_params * 4 / (1024 ** 2)

print(f"\n===== Model =====")
print(f"Total parameters    : {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size          : {param_size_mb:.4f} MB")
assert total_params < 100_000, f"Over parameter limit: {total_params}"

dummy_input = torch.randn(1, 2, 1280).to(device)

if THOP_AVAILABLE:
    flops, _ = thop_profile(model, inputs=(dummy_input,), verbose=False)
    print(f"FLOPs               : {flops/1e6:.2f} MFLOPs")
else:
    flops = 0.0

model.eval()
with torch.no_grad():
    t0 = time.time()
    for _ in range(200):
        _ = model(dummy_input)
    t1 = time.time()
latency_ms = (t1 - t0) / 200 * 1000
print(f"Avg inference latency: {latency_ms:.2f} ms")

if torch.cuda.is_available():
    print(f"Peak GPU memory     : {torch.cuda.max_memory_allocated()/(1024**2):.2f} MB")

# =============================================================================
# Loss — Focal with AFIB upweighted
# =============================================================================

class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)
print(f"\nClass weights : Normal={CLASS_WEIGHTS[0]}  AFIB={CLASS_WEIGHTS[1]}")
print(f"Focal gamma   : {FOCAL_GAMMA}")

criterion = FocalLoss(weight=class_weights, gamma=FOCAL_GAMMA)

# =============================================================================
# Optimizer & scheduler
# =============================================================================

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999)
)
scheduler = build_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)

# =============================================================================
# Training loop
# =============================================================================

best_val_f1      = 0.0
best_epoch       = 0
best_state       = None
patience_counter = 0
history          = []

print(f"\n===== Training (warmup={WARMUP_EPOCHS} epochs, patience={PATIENCE}) =====")

for epoch in range(EPOCHS):

    # ---- train ----
    model.train()
    train_loss  = 0.0
    train_preds = []
    train_trues = []

    for bX, by in train_loader:
        bX = bX.to(device, non_blocking=True)
        by = by.to(device, non_blocking=True)

        if USE_MIXUP:
            bX, ya, yb, lam = mixup_batch(bX, by, MIXUP_ALPHA)
            out  = model(bX)
            loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)
        else:
            out  = model(bX)
            loss = criterion(out, by)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(out, dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_trues.extend(by.cpu().numpy())

    train_loss /= len(train_loader)
    # Note: train accuracy/F1 computed on NATURAL distribution batches now
    train_acc = 100 * np.mean(np.array(train_preds) == np.array(train_trues))
    train_f1  = f1_score(train_trues, train_preds, average="macro", zero_division=0)

    # ---- validate ----
    model.eval()
    val_loss  = 0.0
    val_preds = []
    val_trues = []

    with torch.no_grad():
        for vX, vy in valid_loader:
            vX = vX.to(device, non_blocking=True)
            vy = vy.to(device, non_blocking=True)
            out      = model(vX)
            val_loss += criterion(out, vy).item()
            preds    = torch.argmax(out, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_trues.extend(vy.cpu().numpy())

    val_loss /= len(valid_loader)
    val_acc   = 100 * np.mean(np.array(val_preds) == np.array(val_trues))
    val_f1    = f1_score(val_trues, val_preds, average="macro", zero_division=0)
    per_cls   = f1_score(val_trues, val_preds, average=None,    zero_division=0)

    # per-class recall for validation — key diagnostic
    val_cm         = confusion_matrix(val_trues, val_preds)
    val_normal_rec = val_cm[0,0] / val_cm[0].sum() if val_cm[0].sum() > 0 else 0
    val_afib_rec   = val_cm[1,1] / val_cm[1].sum() if val_cm[1].sum() > 0 else 0

    scheduler.step()
    cur_lr = optimizer.param_groups[0]["lr"]

    print(f"\nEpoch {epoch+1:03d}/{EPOCHS}  LR={cur_lr:.6f}")
    print(f"  train  loss={train_loss:.4f}  acc={train_acc:.2f}%  macroF1={train_f1:.4f}")
    print(f"  valid  loss={val_loss:.4f}  acc={val_acc:.2f}%  macroF1={val_f1:.4f}")
    print(f"  valid  F1:  Normal={per_cls[0]:.4f}  AFIB={per_cls[1]:.4f}")
    print(f"  valid  recall: Normal={val_normal_rec:.4f}  AFIB={val_afib_rec:.4f}")

    history.append({
        "epoch":           epoch + 1,
        "train_loss":      train_loss,
        "train_accuracy":  train_acc,
        "train_f1":        train_f1,
        "valid_loss":      val_loss,
        "valid_accuracy":  val_acc,
        "valid_f1":        val_f1,
        "valid_f1_normal": per_cls[0],
        "valid_f1_afib":   per_cls[1],
        "valid_recall_normal": val_normal_rec,
        "valid_recall_afib":   val_afib_rec,
        "learning_rate":   cur_lr,
    })

    if val_f1 > best_val_f1:
        best_val_f1      = val_f1
        best_epoch       = epoch + 1
        best_state       = copy.deepcopy(model.state_dict())
        torch.save(best_state, MODEL_PATH)
        patience_counter = 0
        print(f"  *** Best model saved  val_macroF1={best_val_f1:.4f} ***")
    else:
        patience_counter += 1
        print(f"  No improvement. Patience {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print("\n===== Early stopping =====")
            break

print(f"\nBest epoch : {best_epoch}  |  Best val macro F1 : {best_val_f1:.4f}")

pd.DataFrame(history).to_csv(os.path.join(RESULTS_DIR, "training_history.csv"), index=False)

# =============================================================================
# Test evaluation
# =============================================================================

model.load_state_dict(best_state)
model.eval()

test_preds    = []
test_trues    = []
misclassified = []

with torch.no_grad():
    for tX, ty in test_loader:
        tX    = tX.to(device, non_blocking=True)
        out   = model(tX)
        probs = torch.softmax(out, dim=1)
        preds = torch.argmax(out, dim=1)

        test_preds.extend(preds.cpu().numpy())
        test_trues.extend(ty.numpy())

        for i in range(len(ty)):
            tl = ty[i].item()
            pl = preds[i].item()
            if tl != pl:
                misclassified.append({
                    "Actual":     LABELS[tl],
                    "Predicted":  LABELS[pl],
                    "Confidence": round(probs[i][pl].item(), 4),
                })

# =============================================================================
# Metrics
# =============================================================================

test_acc     = 100 * np.mean(np.array(test_preds) == np.array(test_trues))
macro_f1     = f1_score(test_trues, test_preds, average="macro",    zero_division=0)
weighted_f1  = f1_score(test_trues, test_preds, average="weighted", zero_division=0)
per_class_f1 = f1_score(test_trues, test_preds, average=None,       zero_division=0)
cm           = confusion_matrix(test_trues, test_preds)
cm_norm      = confusion_matrix(test_trues, test_preds, normalize="true")

# Per-class recall directly from confusion matrix
normal_recall = cm[0,0] / cm[0].sum()
afib_recall   = cm[1,1] / cm[1].sum()
normal_prec   = cm[0,0] / cm[:,0].sum()
afib_prec     = cm[1,1] / cm[:,1].sum()

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)
print(f"  Test Accuracy   : {test_acc:.2f}%")
print(f"  Macro F1        : {macro_f1:.4f}")
print(f"  Weighted F1     : {weighted_f1:.4f}")
print(f"\n  {'Class':8s}  {'Recall':>8}  {'Precision':>10}  {'F1':>8}")
print(f"  {'Normal':8s}  {normal_recall:8.4f}  {normal_prec:10.4f}  {per_class_f1[0]:8.4f}")
print(f"  {'AFIB':8s}  {afib_recall:8.4f}  {afib_prec:10.4f}  {per_class_f1[1]:8.4f}")
print(f"\nConfusion Matrix (rows=True, cols=Predicted):")
print(f"               Pred Normal  Pred AFIB")
print(f"  True Normal  {cm[0,0]:10d}  {cm[0,1]:9d}")
print(f"  True AFIB    {cm[1,0]:10d}  {cm[1,1]:9d}")
print(f"\nNormalized:")
print(np.round(cm_norm, 4))
print(f"\nClassification Report:")
print(classification_report(test_trues, test_preds, target_names=LABELS, zero_division=0))

# =============================================================================
# Save
# =============================================================================

pd.DataFrame(cm).to_csv(os.path.join(RESULTS_DIR, "confusion_matrix.csv"), index=False)
pd.DataFrame(cm_norm).to_csv(os.path.join(RESULTS_DIR, "normalized_confusion_matrix.csv"), index=False)
pd.DataFrame(misclassified).to_csv(os.path.join(RESULTS_DIR, "misclassified_samples.csv"), index=False)
pd.DataFrame([{
    "test_accuracy":    test_acc,
    "macro_f1":         macro_f1,
    "weighted_f1":      weighted_f1,
    "f1_normal":        per_class_f1[0],
    "f1_afib":          per_class_f1[1],
    "recall_normal":    normal_recall,
    "recall_afib":      afib_recall,
    "precision_normal": normal_prec,
    "precision_afib":   afib_prec,
    "params":           total_params,
    "model_size_mb":    param_size_mb,
    "flops_mflops":     flops / 1e6 if flops else 0,
    "latency_ms":       latency_ms,
    "best_epoch":       best_epoch,
    "best_val_f1":      best_val_f1,
}]).to_csv(os.path.join(RESULTS_DIR, "final_metrics.csv"), index=False)

print(f"\nTotal misclassified : {len(misclassified)}")
print(f"Results saved to    : {RESULTS_DIR}")
print("===== Done =====")
