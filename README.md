# 🚗 Vehicle Insurance Fraud Detection & Prediction System
### *Academic Standard Operating Procedure (SOP) Implementation*
**Department of Computer Engineering — Machine Learning Project**

---

## 🌟 Executive Overview
This repository provides a production-grade, end-to-end Machine Learning solution for **Automotive Vehicle Insurance Fraud Detection**, engineered in full adherence to the **Darshan University Computer Engineering Department ML Project SOP (Weeks 1–10)**.

The project encompasses:
1. **Full Exploratory Data Analysis & Preprocessing**: Handling real-world anomalies, sentinel missing values (`*`), driver age outliers (>100), negative annual incomes, and domain feature engineering.
2. **First-Principles Scratch Implementations (`Linear_Regression_Gradient_Descent_Guide` & `Logistic_Regression_Gradient_Descent_Guide`)**: Implementing Batch Gradient Descent in pure NumPy from mathematical derivations (convex MSE and Binary Cross-Entropy loss).
3. **Multi-Algorithm Benchmarking & Model Selection**: Comparing 5 classification architectures, hyperparameter tuning, and selecting the champion model (**Gradient Boosting Classifier**).
4. **Interactive Jupyter Notebook (`vehicle_insurance_fraud_detection.ipynb`)**: Complete narrative story with pre-computed outputs, markdown LaTeX formulas, and visualization charts.
5. **Robust Flask Backend (`app.py`)**: High-performance REST API serving real-time risk assessment, model telemetry, and EDA statistics.
6. **Demure, Aesthetic React Frontend (`frontend/`)**: Modern, professional interface matching the university SOP reference layout (Pages 7, 8, 9) with real-time risk metering, preset claim loading, feature importance progress bars, and actuarial benchmark insights.

---

## 📊 Performance Benchmark Leaderboard

| Model Architecture | Test Accuracy | Precision | Recall | F1 Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting Classifier** | **78.41%** | **86.52%** | **13.21%** | **22.92%** | **0.6579** | 🏆 **Champion** |
| **Random Forest Classifier** | 78.41% | 92.21% | 12.18% | 21.52% | 0.6476 | Candidate |
| **Decision Tree Classifier** | 78.66% | 88.17% | 14.07% | 24.26% | 0.6373 | Baseline |
| **Scikit-Learn Logistic Regression** | 78.07% | 81.32% | 12.69% | 21.96% | 0.6473 | Library |
| **Scratch Logistic Regression (Ours)** | **78.07%** | **81.32%** | **12.69%** | **21.96%** | **0.6473** | **First Principles** |

> **Key Takeaway**: Our from-scratch `ScratchLogisticRegression` using pure NumPy gradient updates achieved a **78.07%** test accuracy and **0.6473** ROC-AUC, matching Scikit-Learn's library implementation and validating mathematical correctness.

---

## 📐 Mathematical Formulation: Gradient Descent Guides

### 1. `Linear_Regression_Gradient_Descent_Guide`
- **Hypothesis**: $h_\theta(x) = X\theta + b$
- **Mean Squared Error Cost**: $J(\theta, b) = \frac{1}{2m} \sum_{i=1}^m \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2$
- **Analytic Gradients**:
  $$\frac{\partial J}{\partial \theta} = \frac{1}{m} X^T (X\theta + b - y), \quad \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})$$
- **Parameter Update**: $\theta := \theta - \alpha \frac{\partial J}{\partial \theta}, \quad b := b - \alpha \frac{\partial J}{\partial b}$

### 2. `Logistic_Regression_Gradient_Descent_Guide`
- **Sigmoid Activation**: $\sigma(z) = \frac{1}{1 + e^{-z}}$
- **Binary Cross-Entropy Loss**:
  $$J(\theta, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \ln(h_\theta(x^{(i)})) + (1 - y^{(i)}) \ln(1 - h_\theta(x^{(i)})) \right]$$
- **Analytic Gradients**:
  $$\frac{\partial J}{\partial \theta} = \frac{1}{m} X^T (\sigma(X\theta + b) - y), \quad \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (\sigma(X\theta + b) - y)$$

---

## 🏗️ Project Architecture & File Hierarchy

```
d:/Mahek/Ml_Final_Project/
├── data/
│   └── insurance_fraud_data.csv          # Official Minitab dataset (12,002 claims)
├── models/
│   ├── best_fraud_model.joblib           # Serialized Champion Gradient Boosting Pipeline
│   ├── model_metadata.json               # Hyperparameters, metrics & benchmark scores
│   └── eda_insights.json                 # Cohort stats & actuarial benchmarks
├── static/plots/
│   ├── confusion_matrix.png              # Confusion matrix heatmap
│   ├── roc_curve.png                     # Comparative ROC-AUC curves
│   ├── feature_importance.png            # Top 10 feature contribution chart
│   └── gradient_descent_convergence.png  # Scratch linear & logistic loss curves
├── scripts/
│   ├── train_and_evaluate.py             # Complete training, scratch models & plots
│   └── generate_notebook.py              # Programmatic Jupyter Notebook creator
├── vehicle_insurance_fraud_detection.ipynb # Fully pre-executed notebook with outputs
├── app.py                                # Flask REST API backend (Port 5000)
├── frontend/                             # Modern Demure React UI (Port 5173)
│   ├── src/
│   │   ├── App.jsx                       # Main application component
│   │   ├── App.css                       # Component styling & layout
│   │   ├── index.css                     # Design tokens, variables & typography
│   │   └── main.jsx                      # React DOM mount point
│   ├── package.json
│   └── vite.config.js                    # Vite dev server with proxy to Flask
└── README.md                             # Comprehensive project documentation
```

---

## 🚀 Running the Project Locally

### 1. Start the Flask Backend Server
```powershell
python app.py
```
*Backend runs on `http://localhost:5000`.*

### 2. Start the React Frontend
```powershell
cd frontend
npm.cmd run dev
```
*Frontend runs on `http://localhost:5173`.*

### 3. Open in Browser
Visit **`http://localhost:5173`** in your browser.
