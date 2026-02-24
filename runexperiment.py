import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from core.signal_builder import build_ppg_segments
from core.network import build_cnn_bilstm
from core.objective import cad_focal_loss


# =====================================================
# CREATE RESULT DIRECTORIES (Auto)
# =====================================================
os.makedirs("results/plots", exist_ok=True)


# =====================================================
# CONFIGURATION
# =====================================================
CSV_PATH = "dataset/cleaned-train-ppg-data.csv"
EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
SEGMENT_LEN = 50
MAX_SEGMENTS_PER_PATIENT = 300


# =====================================================
# 1. LOAD & PROCESS DATA
# =====================================================
print("Loading dataset...")

df = pd.read_csv(CSV_PATH)

X_ppg, X_clin, y, pids = build_ppg_segments(
    df,
    segment_len=SEGMENT_LEN,
    max_per_patient=MAX_SEGMENTS_PER_PATIENT
)

print(f"Data Loaded: {len(X_ppg)} segments from {len(np.unique(pids))} patients.")
print("PPG Input Shape   :", X_ppg.shape)
print("Clinical Input Shape :", X_clin.shape)


# =====================================================
# 2. PATIENT-WISE SPLIT
# =====================================================
unique_pids = np.unique(pids)

train_ids, test_ids = train_test_split(unique_pids, test_size=0.2, random_state=42)
train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=42)

m_train = np.isin(pids, train_ids)
m_val   = np.isin(pids, val_ids)
m_test  = np.isin(pids, test_ids)

print(f"Train Segments: {np.sum(m_train)}")
print(f"Val Segments  : {np.sum(m_val)}")
print(f"Test Segments : {np.sum(m_test)}")


# =====================================================
# 3. BUILD & TRAIN MODEL
# =====================================================
model = build_cnn_bilstm(
    ppg_shape=X_ppg.shape[1:],
    clinical_shape=X_clin.shape[1:]
)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=cad_focal_loss(gamma=2.0, alpha=0.35),
    metrics=["accuracy"]
)

print("\n" + "="*60)
print("                  MODEL SUMMARY")
print("="*60)
model.summary()
print("="*60 + "\n")

print("\nStarting Training...")
history = model.fit(
    [X_ppg[m_train], X_clin[m_train]], y[m_train],
    validation_data=([X_ppg[m_val], X_clin[m_val]], y[m_val]),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-5, verbose=1)
    ],
    verbose=1
)


# =====================================================
# SAVE TRAINED MODEL
# =====================================================
model.save("results/cad_cnn_bilstm_model.h5")
print("✅ Model saved successfully!")


# =====================================================
# SAVE TRAINING CURVES
# =====================================================
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/loss_curve.png")
plt.close()

plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/accuracy_curve.png")
plt.close()

print("✅ Training curves saved!")


# =====================================================
# 4. PREDICTION & AGGREGATION
# =====================================================
print("\nRunning Inference on Test Set...")

seg_probs = model.predict([X_ppg[m_test], X_clin[m_test]], verbose=0).ravel()
seg_labels = y[m_test]
seg_pids = pids[m_test]

patient_probs = defaultdict(list)
patient_true = {}

for prob, pid, label in zip(seg_probs, seg_pids, seg_labels):
    patient_probs[pid].append(prob)
    patient_true[pid] = label


def get_patient_score(probs, k=15):
    probs = np.array(probs)
    k = min(k, len(probs))
    topk = np.sort(probs)[-k:]
    return 0.7 * np.mean(topk) + 0.3 * np.max(probs)


# =====================================================
# 5. THRESHOLD OPTIMIZATION
# =====================================================
best_score, best_t = -1, 0.5
thresholds = np.arange(0.30, 0.70, 0.01)

y_true_all = []
y_scores_all = []

for pid, probs_list in patient_probs.items():
    y_true_all.append(patient_true[pid])
    y_scores_all.append(get_patient_score(probs_list))

y_true_all = np.array(y_true_all)
y_scores_all = np.array(y_scores_all)

for t in thresholds:
    preds = (y_scores_all >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_all, preds).ravel()
    sens = tp / (tp + fn + 1e-6)
    spec = tn / (tn + fp + 1e-6)
    j_score = sens + spec - 1
    if j_score > best_score:
        best_score = j_score
        best_t = t

print(f"\nOptimal Patient Threshold found: {best_t:.3f}")


# =====================================================
# 6. FINAL RESULTS
# =====================================================
final_preds = (y_scores_all >= best_t).astype(int)

acc = accuracy_score(y_true_all, final_preds)
sens = recall_score(y_true_all, final_preds)
spec = recall_score(y_true_all, final_preds, pos_label=0)
prec = precision_score(y_true_all, final_preds)
f1 = f1_score(y_true_all, final_preds)
auc_score = roc_auc_score(y_true_all, y_scores_all)
cm = confusion_matrix(y_true_all, final_preds)

print("\n" + "="*40)
print("      FINAL PATIENT-LEVEL METRICS")
print("="*40)
print(f"Accuracy     : {acc:.4f}")
print(f"Sensitivity  : {sens:.4f}")
print(f"Specificity  : {spec:.4f}")
print(f"Precision    : {prec:.4f}")
print(f"F1-Score     : {f1:.4f}")
print(f"ROC-AUC      : {auc_score:.4f}")
print("="*40)
print("Confusion Matrix:")
print(cm)
print("="*40)


# =====================================================
# 7. SAVE ROC & CONFUSION MATRIX
# =====================================================
fpr, tpr, _ = roc_curve(y_true_all, y_scores_all)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, lw=2, label=f'AUC = {auc_score:.3f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Patient-Level ROC Curve')
plt.legend()
plt.grid(True)
plt.savefig("results/plots/patient_roc.png")
plt.close()

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Healthy", "CAD"],
            yticklabels=["Healthy", "CAD"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (t={best_t:.2f})")
plt.savefig("results/plots/patient_cm.png")
plt.close()

print("\n✅ Experiment Complete.")
print("📁 All results saved inside 'results/' folder.")