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
all_ann_pred = []
all_log_pred = []
all_base_pred = []
all_y_true = []

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

    # Store predictions for statistical evaluation
    all_ann_pred.extend(ann_pred)
    all_log_pred.extend(log_pred)
    all_base_pred.extend(baseline_pred)
    all_y_true.extend(y_test)

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

# Complete Statistical Evaluation with McNemar's Test for all pairs
def create_contingency_table(pred1, pred2, true):
    table = np.array([[0, 0], [0, 0]])
    for i in range(len(true)):
        if pred1[i] == true[i] and pred2[i] != true[i]:
            table[0, 1] += 1  # pred1 correct, pred2 incorrect
        elif pred1[i] != true[i] and pred2[i] == true[i]:
            table[1, 0] += 1  # pred1 incorrect, pred2 correct
    return table

# Convert lists to arrays
all_ann_pred = np.array(all_ann_pred)
all_log_pred = np.array(all_log_pred)
all_base_pred = np.array(all_base_pred)
all_y_true = np.array(all_y_true)

# Perform pairwise comparisons
print("\nStatistical Evaluation (McNemar’s Test):")

# 1. ANN vs Logistic Regression
table_ann_log = create_contingency_table(all_ann_pred, all_log_pred, all_y_true)
mcnemar_ann_log = mcnemar(table_ann_log, exact=True)
print("\nANN vs Logistic Regression:")
print(f"Contingency Table:\n{table_ann_log}")
print(f"Test Statistic: {mcnemar_ann_log.statistic}")
print(f"P-Value: {mcnemar_ann_log.pvalue}")

# 2. ANN vs Baseline
table_ann_base = create_contingency_table(all_ann_pred, all_base_pred, all_y_true)
mcnemar_ann_base = mcnemar(table_ann_base, exact=True)
print("\nANN vs Baseline:")
print(f"Contingency Table:\n{table_ann_base}")
print(f"Test Statistic: {mcnemar_ann_base.statistic}")
print(f"P-Value: {mcnemar_ann_base.pvalue}")

# 3. Logistic Regression vs Baseline
table_log_base = create_contingency_table(all_log_pred, all_base_pred, all_y_true)
mcnemar_log_base = mcnemar(table_log_base, exact=True)
print("\nLogistic Regression vs Baseline:")
print(f"Contingency Table:\n{table_log_base}")
print(f"Test Statistic: {mcnemar_log_base.statistic}")
print(f"P-Value: {mcnemar_log_base.pvalue}")

print("\nScript completed successfully!")

table = pd.DataFrame(
    {
        'Outer fold': np.arange(1, K + 1),
        'ANN n': h_list.flatten(),
        'ANN E': np.round(Error_test_nn,10).flatten(),
        'Linear regression lambda': found_lambdas.flatten(),
        'Linear regression E': np.round(Error_test_rlr,10).flatten(),
        'Baseline E': np.round(Error_test_nofeatures,10).flatten()
    }
)

print(table)

table.to_csv('regression_table.csv', index=False)