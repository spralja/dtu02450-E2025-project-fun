import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from statsmodels.stats.contingency_tables import mcnemar
from collections import Counter
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Load dataset
filename = 'data/glass+identification/glass.csv'
data = pd.read_csv(filename)

# Extract attributes and target
attributeNames = np.asarray(data.columns)
rawvalues = data.values
X = rawvalues[:, 1:-1]  # Exclude ID and Type
y = rawvalues[:, -1]    # Target (Type)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Two-level CV setup
outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
lambda_values = [0.001, 0.01, 0.1, 1, 10]
h_values = [1, 3, 5, 10]
results = []

# Perform two-level cross-validation
for i, (train_idx, test_idx) in enumerate(outer_cv.split(X_scaled)):
    print(f"Processing outer fold {i+1}/10")
    X_par, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_par, y_test = y[train_idx], y[test_idx]

    # Inner CV for ANN
    best_h, best_ann_err = None, float('inf')
    for h in h_values:
        ann = MLPClassifier(hidden_layer_sizes=(h,), activation='relu', 
                            max_iter=5000, learning_rate_init=0.01, random_state=42)
        ann_err = 0
        for inner_train_idx, val_idx in inner_cv.split(X_par):
            ann.fit(X_par[inner_train_idx], y_par[inner_train_idx])
            pred = ann.predict(X_par[val_idx])
            ann_err += np.mean(pred != y_par[val_idx])
        ann_err /= 5
        if ann_err < best_ann_err:
            best_h, best_ann_err = h, ann_err

    # Inner CV for Logistic Regression
    best_lambda, best_log_err = None, float('inf')
    for lam in lambda_values:
        log_reg = LogisticRegression(C=1/lam, max_iter=1000, random_state=42)
        log_err = 0
        for inner_train_idx, val_idx in inner_cv.split(X_par):
            log_reg.fit(X_par[inner_train_idx], y_par[inner_train_idx])
            pred = log_reg.predict(X_par[val_idx])
            log_err += np.mean(pred != y_par[val_idx])
        log_err /= 5
        if log_err < best_log_err:
            best_lambda, best_log_err = lam, log_err

    # Train and test on outer fold
    ann = MLPClassifier(hidden_layer_sizes=(best_h,), activation='relu', 
                        max_iter=5000, learning_rate_init=0.01, random_state=42)
    log_reg = LogisticRegression(C=1/best_lambda, max_iter=1000, random_state=42)
    ann.fit(X_par, y_par)
    log_reg.fit(X_par, y_par)
    baseline_pred = np.full_like(y_test, Counter(y_par).most_common(1)[0][0])

    ann_pred = ann.predict(X_test)
    log_pred = log_reg.predict(X_test)

    ann_err = np.mean(ann_pred != y_test)
    log_err = np.mean(log_pred != y_test)
    base_err = np.mean(baseline_pred != y_test)

    results.append([i + 1, best_h, ann_err, best_lambda, log_err, base_err])

# Create Table 2
columns = ['Outer Fold', 'ANN h*', 'ANN E_test', 'LogReg λ*', 'LogReg E_test', 'Baseline E_test']
table = pd.DataFrame(results, columns=columns)
print("\nTable 2: Two-Level Cross-Validation Results")
print(table.to_string(index=False))

# Summary statistics
print("\nSummary Statistics:")
print(f"Average ANN Error: {np.mean(table['ANN E_test']):.4f}")
print(f"Average Logistic Regression Error: {np.mean(table['LogReg E_test']):.4f}")
print(f"Average Baseline Error: {np.mean(table['Baseline E_test']):.4f}")

# McNemar’s Test for ANN vs Logistic Regression
contingency_table = np.array([[0, 0], [0, 0]])
for i in range(len(y_test)):
    if ann_pred[i] == y_test[i] and log_pred[i] != y_test[i]:
        contingency_table[0, 1] += 1  # ANN correct, LogReg incorrect
    elif ann_pred[i] != y_test[i] and log_pred[i] == y_test[i]:
        contingency_table[1, 0] += 1  # ANN incorrect, LogReg correct

mcnemar_result = mcnemar(contingency_table, exact=True)
print("\nStatistical Evaluation (McNemar’s Test):")
print(f"Test Statistic: {mcnemar_result.statistic}")
print(f"P-Value: {mcnemar_result.pvalue}")

print("\nScript completed successfully!")