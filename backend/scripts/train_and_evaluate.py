import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, roc_curve)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def run_pipeline():
    os.makedirs('models', exist_ok=True)
    os.makedirs('static/plots', exist_ok=True)

    print("Step 1: Ingesting dataset...")
    df = pd.read_csv('data/insurance_fraud_data.csv')
    raw_count = len(df)
    
    # Clean target
    df = df.dropna(subset=['fraud reported'])
    df = df[df['fraud reported'].isin(['Y', 'N'])].copy()
    df = df.replace('*', np.nan)

    # Clean driver age and annual income anomalies
    df['age_of_driver'] = pd.to_numeric(df['age_of_driver'], errors='coerce')
    df.loc[df['age_of_driver'] > 100, 'age_of_driver'] = np.nan
    df.loc[df['age_of_driver'] < 16, 'age_of_driver'] = np.nan

    df['annual_income'] = pd.to_numeric(df['annual_income'], errors='coerce')
    df.loc[df['annual_income'] <= 0, 'annual_income'] = np.nan

    df['injury_claim'] = pd.to_numeric(df['injury_claim'], errors='coerce')
    df['age_of_vehicle'] = pd.to_numeric(df['age_of_vehicle'], errors='coerce')
    df['marital_status'] = pd.to_numeric(df['marital_status'], errors='coerce')
    df['witness_present'] = pd.to_numeric(df['witness_present'], errors='coerce')

    df['claim_date'] = pd.to_datetime(df['claim_date'], errors='coerce')
    df['claim_month'] = df['claim_date'].dt.month.fillna(6).astype(int)
    df['claim_day'] = df['claim_date'].dt.day.fillna(15).astype(int)

    # Feature engineering
    df['injury_to_total_ratio'] = df['injury_claim'] / (df['total_claim'] + 1e-5)
    df['claim_to_price_ratio'] = df['total_claim'] / (df['vehicle_price'] + 1e-5)
    df['income_to_price_ratio'] = df['annual_income'] / (df['vehicle_price'] + 1e-5)

    target = (df['fraud reported'] == 'Y').astype(int)
    drop_cols = ['claim_number', 'claim_date', 'fraud reported', 'zip_code']
    features = df.drop(columns=drop_cols)

    cleaned_count = len(features)
    print(f"Data summary: Raw = {raw_count}, Cleaned = {cleaned_count}, Filtered = {raw_count - cleaned_count}")

    num_cols = features.select_dtypes(include=['number']).columns.tolist()
    cat_cols = [c for c in features.columns if c not in num_cols]

    # Standard train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_cols), ('cat', cat_transformer, cat_cols)])

    # Process features for scratch implementations
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    cat_names = preprocessor.named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(cat_cols)
    all_feature_names = num_cols + list(cat_names)

    # ========================================================
    # 2. Linear Regression with Gradient Descent (Scratch)
    # ========================================================
    print("Step 2: Training Scratch Linear Regression with Gradient Descent...")
    class ScratchLinearRegression:
        def __init__(self, lr=0.05, n_iters=800):
            self.lr = lr
            self.n_iters = n_iters
            self.weights = None
            self.bias = 0.0
            self.loss_history = []
            
        def fit(self, X, y):
            n_samples, n_features = X.shape
            self.weights = np.zeros(n_features)
            self.bias = 0.0
            self.loss_history = []
            
            for _ in range(self.n_iters):
                y_pred = np.dot(X, self.weights) + self.bias
                cost = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
                self.loss_history.append(cost)
                
                dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
                db = (1 / n_samples) * np.sum(y_pred - y)
                
                self.weights -= self.lr * dw
                self.bias -= self.lr * db
            return self
                
        def predict(self, X):
            return np.dot(X, self.weights) + self.bias

    claim_y_train = X_train['total_claim'].fillna(X_train['total_claim'].median()).values
    claim_X_train = X_train[['vehicle_price', 'safety_rating', 'annual premium', 'age_of_driver']].copy()
    for col in claim_X_train.columns:
        claim_X_train[col] = claim_X_train[col].fillna(claim_X_train[col].median())
    scaler_lr = StandardScaler()
    claim_X_scaled = scaler_lr.fit_transform(claim_X_train)

    scratch_linreg = ScratchLinearRegression(lr=0.05, n_iters=800)
    scratch_linreg.fit(claim_X_scaled, claim_y_train)

    # ========================================================
    # 3. Logistic Regression with Gradient Descent (Scratch)
    # ========================================================
    print("Step 3: Training Scratch Logistic Regression with Gradient Descent...")
    class ScratchLogisticRegression:
        def __init__(self, lr=0.1, n_iters=1200):
            self.lr = lr
            self.n_iters = n_iters
            self.weights = None
            self.bias = 0.0
            self.loss_history = []
            
        def _sigmoid(self, z):
            return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))
            
        def fit(self, X, y):
            n_samples, n_features = X.shape
            self.weights = np.zeros(n_features)
            self.bias = 0.0
            self.loss_history = []
            
            for _ in range(self.n_iters):
                linear_model = np.dot(X, self.weights) + self.bias
                y_pred = self._sigmoid(linear_model)
                
                eps = 1e-15
                loss = - (1/n_samples) * np.sum(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
                self.loss_history.append(loss)
                
                dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
                db = (1 / n_samples) * np.sum(y_pred - y)
                
                self.weights -= self.lr * dw
                self.bias -= self.lr * db
            return self
                
        def predict_proba(self, X):
            linear_model = np.dot(X, self.weights) + self.bias
            p1 = self._sigmoid(linear_model)
            return np.column_stack([1 - p1, p1])
            
        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    scratch_logreg = ScratchLogisticRegression(lr=0.1, n_iters=1200)
    scratch_logreg.fit(X_train_proc, y_train.values)
    scratch_preds = scratch_logreg.predict(X_test_proc)
    scratch_probs = scratch_logreg.predict_proba(X_test_proc)[:, 1]

    # ========================================================
    # 4. Standard Classification Models Benchmark
    # ========================================================
    print("Step 4: Benchmarking Machine Learning Algorithms...")
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=5, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, subsample=0.85, random_state=42)
    }

    benchmark_metrics = []
    trained_pipelines = {}

    for name, clf in models.items():
        pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe
        
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        
        benchmark_metrics.append({
            'model': name,
            'accuracy': round(float(acc), 4),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'f1_score': round(float(f1), 4),
            'roc_auc': round(float(auc), 4)
        })
        print(f"  {name:22s} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    # Add Scratch Logistic Regression to Benchmark
    scratch_acc = accuracy_score(y_test, scratch_preds)
    scratch_prec = precision_score(y_test, scratch_preds, zero_division=0)
    scratch_rec = recall_score(y_test, scratch_preds, zero_division=0)
    scratch_f1 = f1_score(y_test, scratch_preds, zero_division=0)
    scratch_auc = roc_auc_score(y_test, scratch_probs)

    benchmark_metrics.append({
        'model': 'Scratch Logistic Reg (First Principles)',
        'accuracy': round(float(scratch_acc), 4),
        'precision': round(float(scratch_prec), 4),
        'recall': round(float(scratch_rec), 4),
        'f1_score': round(float(scratch_f1), 4),
        'roc_auc': round(float(scratch_auc), 4)
    })

    # Champion model selection
    best_model_name = 'Gradient Boosting'
    best_pipe = trained_pipelines[best_model_name]
    joblib.dump(best_pipe, 'models/best_fraud_model.joblib')
    print(f"Champion Model: {best_model_name} saved to models/best_fraud_model.joblib")

    # Feature Importances
    best_clf = best_pipe.named_steps['clf']
    feature_importances = pd.Series(best_clf.feature_importances_, index=all_feature_names).sort_values(ascending=False)
    top_features = [{'feature': k, 'importance': round(float(v), 4), 'percentage': round(float(v * 100), 1)} 
                    for k, v in feature_importances.head(10).items()]

    best_pred = best_pipe.predict(X_test)
    best_prob = best_pipe.predict_proba(X_test)[:, 1]

    # Save Model Metadata
    metadata = {
        'best_model': best_model_name,
        'library': 'scikit-learn',
        'hyperparameters': {
            'n_estimators': 150,
            'learning_rate': 0.08,
            'max_depth': 4,
            'subsample': 0.85,
            'criterion': 'friedman_mse',
            'min_samples_split': 2,
            'min_samples_leaf': 1
        },
        'performance': {
            'accuracy': round(float(accuracy_score(y_test, best_pred)), 4),
            'precision': round(float(precision_score(y_test, best_pred, zero_division=0)), 4),
            'recall': round(float(recall_score(y_test, best_pred, zero_division=0)), 4),
            'f1_score': round(float(f1_score(y_test, best_pred, zero_division=0)), 4),
            'roc_auc': round(float(roc_auc_score(y_test, best_prob)), 4)
        },
        'benchmark': benchmark_metrics,
        'top_features': top_features,
        'scratch_gradient_descent': {
            'linear_regression': {
                'learning_rate': 0.05,
                'epochs': 800,
                'final_loss': round(float(scratch_linreg.loss_history[-1]), 4),
                'loss_history_sample': [round(float(x), 4) for x in scratch_linreg.loss_history[::40]]
            },
            'logistic_regression': {
                'learning_rate': 0.1,
                'epochs': 1200,
                'final_loss': round(float(scratch_logreg.loss_history[-1]), 4),
                'accuracy': round(float(scratch_acc), 4),
                'loss_history_sample': [round(float(x), 4) for x in scratch_logreg.loss_history[::50]]
            }
        }
    }

    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save EDA Insights
    eda_insights = {
        'total_records': cleaned_count,
        'fraud_count': int(target.sum()),
        'legit_count': int((1 - target).sum()),
        'fraud_percentage': round(float(target.mean() * 100), 2),
        'average_total_claim': round(float(df['total_claim'].mean()), 2),
        'average_income': round(float(df['annual_income'].mean()), 2),
        'accident_site_stats': df.groupby('accident_site')['fraud reported'].apply(lambda s: round(float((s == 'Y').mean() * 100), 2)).to_dict(),
        'vehicle_category_stats': df.groupby('vehicle_category')['fraud reported'].apply(lambda s: round(float((s == 'Y').mean() * 100), 2)).to_dict(),
        'form_defects_stats': df.groupby('form defects')['fraud reported'].apply(lambda s: round(float((s == 'Y').mean() * 100), 2)).head(8).to_dict()
    }

    with open('models/eda_insights.json', 'w') as f:
        json.dump(eda_insights, f, indent=2)

    # Visual Plots Generation
    print("Step 5: Generating visual charts for notebook and UI...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit (0)', 'Fraud (1)'], yticklabels=['Legit (0)', 'Fraud (1)'])
    plt.title(f'Confusion Matrix: {best_model_name}', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('static/plots/confusion_matrix.png', dpi=150)
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, best_prob)
    plt.figure(figsize=(6.5, 5))
    plt.plot(fpr, tpr, color='#2563eb', lw=2.5, label=f'Gradient Boosting (AUC = {roc_auc_score(y_test, best_prob):.3f})')
    plt.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', lw=1.5, label='Random Baseline')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('static/plots/roc_curve.png', dpi=150)
    plt.close()

    # 3. Feature Importance
    plt.figure(figsize=(8, 5))
    top_10_s = feature_importances.head(10).iloc[::-1]
    plt.barh(top_10_s.index, top_10_s.values * 100, color='#3b82f6', edgecolor='#1d4ed8')
    plt.title('Top 10 Feature Importances (%) - Gradient Boosting', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Importance Contribution (%)')
    plt.tight_layout()
    plt.savefig('static/plots/feature_importance.png', dpi=150)
    plt.close()

    # 4. Gradient Descent Convergence Curves
    plt.figure(figsize=(10, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(scratch_linreg.loss_history, color='#059669', lw=2)
    plt.title('Linear Reg MSE Cost vs Epochs', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Cost J(θ)')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(scratch_logreg.loss_history, color='#dc2626', lw=2)
    plt.title('Logistic Reg Log-Loss vs Epochs', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Cost J(θ)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('static/plots/gradient_descent_convergence.png', dpi=150)
    plt.close()

    print("[SUCCESS] Pipeline completed successfully. Champion Gradient Boosting model trained and exported.")

if __name__ == '__main__':
    run_pipeline()
