# 🫀 Coronary Artery Disease Detection Using PPG Signals

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Model](https://img.shields.io/badge/Architecture-CNN%20%2B%20BiLSTM-green)
![AUC](https://img.shields.io/badge/AUC-0.944-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📌 Project Overview

This project presents a deep learning-based system for detecting **Coronary Artery Disease (CAD)** using **Photoplethysmography (PPG) signals**.

PPG is a non-invasive and cost-effective biomedical signal that measures blood volume variations in peripheral circulation.  
The objective of this work is to design an intelligent system that classifies patients into:

- ✅ Healthy  
- ❌ CAD (Coronary Artery Disease)

The system performs **patient-level classification** using a hybrid deep learning model that captures both spatial and temporal characteristics of PPG waveforms.

---

## 🎯 Objectives

- Preprocess raw PPG signal data  
- Extract meaningful waveform representations  
- Develop a hybrid CNN + BiLSTM architecture  
- Handle class imbalance using Focal Loss  
- Perform patient-level aggregation  
- Evaluate performance using clinically relevant metrics  

---

## 3️⃣ Dataset Description

The dataset consists of PPG signals and clinical features stored in CSV format. Each record includes:

- **SUBJECT_ID** (Unique patient ID)  
- **CAD_LABEL** (0 = Healthy, 1 = CAD)  
- **AGE**  
- **GENDER**  
- **is_diabetic**  
- **has_high_cholesterol**  
- **is_obese**  
- **Raw PPG waveform values**  

The dataset represents realistic hospital conditions and allows robust evaluation of model performance.

---

## 4️⃣ Data Preprocessing

### 4.1 PPG Parsing
PPG signals stored as strings are converted into numerical arrays using custom parsing functions.

### 4.2 Signal Segmentation
- Each continuous PPG signal is divided into fixed-length segments  
- **50 samples per segment**  
- Segments with near-zero standard deviation are removed  

### 4.3 Segment Limitation
- Maximum **300 segments per patient**  
- Prevents bias from highly sampled patients  

### 4.4 Clinical Feature Extraction
Clinical features used:
- Age  
- Gender  
- Diabetes  
- High cholesterol  
- Obesity  

These features are fused with signal-based features for improved prediction.

---

## 5️⃣ Proposed Model Architecture

The model consists of two branches:

### 5.1 PPG Branch – CNN + BiLSTM + Attention

#### 🔹 Convolutional Neural Network (CNN)

- Conv1D (32 filters, kernel=5)  
- Conv1D (64 filters, kernel=5)  
- Conv1D (128 filters, kernel=3)  
- Batch Normalization  
- MaxPooling  

**Purpose:** Extract morphological waveform features such as peaks, slopes, and amplitude variations.

#### 🔹 Bidirectional LSTM

- 32 LSTM units  
- Bidirectional processing  
- Dropout = 0.3  
- Recurrent dropout = 0.2  

**Purpose:** Capture temporal dependencies in heartbeat sequences.

#### 🔹 Attention Pooling Layer

- Learns weights for each time step  
- Applies softmax normalization  
- Produces weighted feature representation  

**Purpose:** Focus on diagnostically important signal segments.

---

### 5.2 Clinical Branch

- Dense(16, ReLU)  
- Dropout(0.3)  

**Purpose:** Learn interactions between clinical risk factors.

---

### 5.3 Feature Fusion

- Concatenation of signal and clinical features  
- Dense(64, ReLU)  
- Dropout(0.4)  
- Final Dense(1, Sigmoid)  

**Output:** Probability of CAD presence.

---

## 6️⃣ Training Strategy

### Optimizer
- Adam  
- Learning rate = 3e-4  

### Loss Function
**Focal Loss**
- gamma = 2.0  
- alpha = 0.35  

**Purpose:** Handle class imbalance and focus on hard examples.

### Training Configuration
- Epochs = 40  
- Batch size = 64  
- EarlyStopping (patience=8)  
- ReduceLROnPlateau (patience=4)  

Training curves are saved for analysis.

---

## 7️⃣ Patient-Level Aggregation Strategy

Instead of predicting CAD per segment, predictions are aggregated per patient.

### 🔹 Top-K Hybrid Strategy

For each patient:
- Select top 15 highest probability segments  
- Compute:

```
Score = 0.7 × Mean(Top-K) + 0.3 × Max(segment probability)
```

Ensures:
- Abnormal bursts are not ignored  
- Noise does not dominate prediction  

---

## 8️⃣ Threshold Optimization

Threshold optimized using **Youden’s J statistic**:

```
J = Sensitivity + Specificity − 1
```

Optimal threshold found:

**0.450**

This ensures the best balance between sensitivity and specificity.

---
Framework:
- TensorFlow / Keras  

---

## ⚙️ Training Configuration

- Optimizer: Adam  
- Learning Rate: Default adaptive learning  
- Loss Function: Focal Loss  
- Batch Size: Configurable  
- Evaluation: Patient-level aggregation  

---

## 📊 Results & Performance

### 🔹 Patient-Level Metrics

| Metric | Value |
|--------|--------|
| Accuracy | ~87% |
| Precision (CAD) | ~85.5% |
| Recall / Sensitivity | ~87.6% |
| Specificity | ~86.5% |
| F1-Score | ~86.5% |
| ROC-AUC Score | **0.944** |
| False Positive Rate (FPR) | ~13.5% |
| False Negative Rate (FNR) | ~12.4% |

---

### 🔹 Confusion Matrix Summary

|                | Predicted Healthy | Predicted CAD |
|---------------|------------------|---------------|
| **Actual Healthy** | 115 | 18 |
| **Actual CAD**     | 15  | 106 |

- True Positives (TP) = 106  
- True Negatives (TN) = 115  
- False Positives (FP) = 18  
- False Negatives (FN) = 15  

---

### 🔹 Interpretation

- High **ROC-AUC (0.944)** indicates excellent discriminative capability.  
- Balanced **Sensitivity (87.6%)** and **Specificity (86.5%)** ensure reliable CAD screening.  
- Low False Negative Rate (~12%) reduces the risk of missing CAD patients.  
- Strong F1-score confirms balanced precision and recall performance.  

## 📈 Performance Visualizations

The repository includes:

- Training vs Validation Accuracy Curve  
- Training vs Validation Loss Curve  
- Confusion Matrix  
- ROC Curve  

All result plots are available inside:

```
results/plots/
```

---

## 📂 Project Structure

```
CAD-Detection-Using-PPG/
│
├── core/
│   ├── network.py
│   ├── objective.py
│   ├── signal_builder.py
│
├── results/
│   └── plots/
│       ├── accuracy_curve.png
│       ├── loss_curve.png
│       ├── patient_cm.png
│       └── patient_roc.png
│
├── runexperiment.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/rohith-1008/CAD-Detection-Using-PPG.git
cd CAD-Detection-Using-PPG
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
python runexperiment.py
```

This will:

- Train the CNN + BiLSTM model  
- Evaluate patient-level performance  
- Generate evaluation plots  

---

## 🔬 Evaluation Metrics

- Accuracy  
- Precision  
- Recall (Sensitivity)  
- Specificity  
- Confusion Matrix  
- ROC Curve  
- AUC Score  

---

## 🔁 Reproducibility

To reproduce the results:

1. Place the processed dataset in the appropriate directory  
2. Install dependencies  
3. Run `runexperiment.py`  

Ensure patient-level separation during training/testing to avoid data leakage.

---

## 🚀 Future Improvements

- Cross-validation experiments  
- Ensemble deep learning approaches  
- Hyperparameter optimization  
- Real-time PPG-based CAD screening system  
- Deployment as a clinical decision support tool  

---

## 🎓 Academic Context

Developed as a Final Year B.Tech Major Project in:

- Artificial Intelligence  
- Biomedical Signal Processing  
- Healthcare Analytics  

---
