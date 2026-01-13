# 🫀 Heart Disease Prediction

> A machine learning classification project to predict the presence of heart disease using clinical health metrics. Implements and compares multiple algorithms to identify the most effective model for heart disease detection.

---

## 🎯 Overview

This project develops a machine learning model to predict the presence of heart disease in patients based on clinical and physiological features. The analysis includes:

* **Data Preprocessing**: Cleaning and preparing medical data for model training
* **Algorithm Comparison**: Evaluating multiple classification algorithms
* **Model Optimization**: Identifying the best-performing model
* **Performance Evaluation**: Comprehensive metrics and visualizations

**Business Impact**: Early detection of heart disease risk enables preventive healthcare interventions, improving patient outcomes and reducing healthcare costs.

---

## 📊 Dataset

### Data Source
* **Dataset**: Heart Attack Medical Data
* **Total Records**: 1,319 patients
* **Features**: 8 clinical variables
* **Target Variable**: Heart disease presence (Binary: 0 = No disease, 1 = Disease)

### Data Quality
* Missing values: Forward fill method applied
* 1,319 complete patient records after preprocessing
* Balanced target variable distribution

---

## 🔍 Methodology

### Machine Learning Pipeline

```
Raw Data → Data Cleaning → Feature Standardization → Train-Test Split → 
Model Training → Prediction → Evaluation
```

### 1. Data Preprocessing
* **Missing Value Handling**: Forward fill method
* **Feature Scaling**: StandardScaler (mean=0, std=1)
  * Normalizes all features to comparable ranges
  * Improves algorithm convergence
* **Train-Test Split**: 70% training, 30% testing (random_state=42)

### 2. Algorithms Evaluated

Three classification algorithms were tested and compared:

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| K-Nearest Neighbors (K=5) | 62.9% | 0.53 | 0.53 | 0.53 |
| Naive Bayes (Gaussian) | 91.4% | 0.83 | 0.99 | 0.90 |
| **Random Forest** | **97.98%** | **0.98** | **0.99** | **0.98** |

### 3. Selected Model: Random Forest Classifier

**Why Random Forest?**
* Highest accuracy (97.98%)
* Excellent precision (0.98) - minimizes false positives
* Excellent recall (0.99) - catches 99% of actual disease cases
* Handles non-linear relationships effectively
* Robust to outliers

---

## 📈 Results

### Best Model Performance (Random Forest)

**Overall Accuracy**: 97.98%
* Model correctly predicts disease presence 98 times out of 100

**Confusion Matrix** (30% Test Set = 396 samples):
```
                Predicted Negative    Predicted Positive
Actual Negative         150                    5
Actual Positive          3                   238
```

**Interpretation:**
* **True Negatives (150)**: Correctly identified 150 healthy patients
* **True Positives (238)**: Correctly identified 238 patients with disease
* **False Positives (5)**: Only 3.2% false alarm rate
* **False Negatives (3)**: Only 1.2% missed cases (critical strength)

### Performance Metrics

```
Metric              Value    Meaning
─────────────────────────────────────────────────
Accuracy            97.98%   Overall correctness
Precision           0.98     Right 98% of time when predicting disease
Recall/Sensitivity  0.99     Catch 99% of actual disease cases
Specificity         0.97     Correctly identify 97% of healthy patients
F1-Score            0.98     Best balance of precision and recall
```


---

## 💾 Installation

### Prerequisites
* Python 3.8 or higher

### Setup Steps

**1. Clone the Repository**
```bash
git clone https://github.com/poojitha1114/heart-disease-prediction.git
cd heart-disease-prediction
```

**2. Install Required Libraries**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

**3. Run the Notebook**
```bash
jupyter notebook cardioascular_disease.ipynb
```

### Required Libraries
* `pandas` - Data manipulation and analysis
* `numpy` - Numerical computing
* `scikit-learn` - Machine learning algorithms
* `matplotlib` - Data visualization
* `seaborn` - Statistical visualization

---

## 🚀 Usage

### Running the Complete Analysis

**1. Open the Jupyter Notebook**
* Launch the `cardioascular_disease.ipynb` notebook
* Run cells sequentially from top to bottom

**2. Step-by-Step Process**
* **Cell 1-2**: Import libraries and load data
* **Cell 3**: Data preprocessing (missing value handling)
* **Cell 4-5**: Feature scaling and train-test split
* **Cell 6-7**: Define and train three models
* **Cell 8-9**: Generate predictions and evaluate results
* **Cell 10**: Visualize feature distributions
* **Cell 11**: Display confusion matrix for best model

---

## 💡 Key Findings

### 1. Random Forest Significantly Outperforms Other Algorithms
* Random Forest: 97.98% accuracy
* Naive Bayes: 91.4% accuracy (7.5% improvement gap)
* KNN: 62.9% accuracy (35% improvement)

**Implication**: Ensemble methods like Random Forest better capture complex relationships in medical data.

### 2. Excellent Disease Detection Rate (99% Recall)
* Only 3 out of 238 actual disease cases were missed
* This is critical in healthcare—missing disease is more dangerous than false alarms
* The model is suitable for preliminary screening

### 3. Low False Positive Rate (3.2%)
* Only 5 out of 155 healthy patients flagged as diseased
* Reduces unnecessary anxiety and follow-up testing
* Cost-effective screening tool

### 4. Feature Importance (Based on Model Analysis)
* **Heart rate (impulse)** - Most discriminative feature
* **Blood pressure** - Strong indicator of cardiovascular stress
* **Troponin levels** - Direct cardiac damage marker
* **Glucose & Cholesterol** - Metabolic risk factors

### 5. Clinical Applicability
The 97.98% accuracy suggests this model could serve as:
* **Preliminary screening tool** for high-risk populations
* **Decision support system** in clinical settings
* **Population health monitoring** platform

---

## 📊 Performance Summary

| Aspect | Value | Assessment |
|--------|-------|-----------|
| Overall Accuracy | 97.98% | Excellent |
| Recall (Sensitivity) | 0.99 | Excellent - Catches diseases |
| Precision | 0.98 | Excellent - Minimal false alarms |
| Specificity | 0.97 | Excellent - Identifies healthy |
| F1-Score | 0.98 | Excellent - Best balance |
| Model Type | Random Forest | Proven effective |
| Training Samples | 923 | Adequate |
| Testing Samples | 396 | Adequate |

---
