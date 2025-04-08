#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%%
#Importing the data
data = pd.read_csv('regression_table.csv')


#%%
print(data)
ANN_E = data['ANN E']
ANN_E = ANN_E.to_numpy()
ANN_E = ANN_E.reshape(-1, 1)
ANN_E = ANN_E.astype(np.float64)

Regression_E = data['Linear regression E']
Regression_E = Regression_E.to_numpy()
Regression_E = Regression_E.reshape(-1, 1)
Regression_E = Regression_E.astype(np.float64)

Baseline_E = data['Baseline E']
Baseline_E = Baseline_E.to_numpy()
Baseline_E = Baseline_E.reshape(-1, 1)
Baseline_E = Baseline_E.astype(np.float64)
#%%
#Comparing models

#taking difference

ANN_Linear_diff = ANN_E - Regression_E # Positive if linear is better
ANN_Baseline_diff = ANN_E - Baseline_E # Positive if baseline is better
Linear_Baseline_diff = Regression_E - Baseline_E # Positive if baseline is better


ANN_Lin_diff_mean = np.mean(ANN_Linear_diff)
ANN_Lin_diff_std = np.std(ANN_Linear_diff)
Lin_Baseline_diff_mean = np.mean(Linear_Baseline_diff)
Lin_Baseline_diff_std = np.std(Linear_Baseline_diff)
ANN_Baseline_diff_mean = np.mean(ANN_Baseline_diff)
ANN_Baseline_diff_std = np.std(ANN_Baseline_diff)

#%%
print("ANN vs Linear regression")
print("Mean: ", ANN_Lin_diff_mean)
print("Standard deviation: ", ANN_Lin_diff_std)
print("\n")
print("ANN vs Baseline")
print("Mean: ", ANN_Baseline_diff_mean)
print("Standard deviation: ", ANN_Baseline_diff_std)
print("\n")
print("Linear regression vs Baseline")
print("Mean: ", Lin_Baseline_diff_mean)
print("Standard deviation: ", Lin_Baseline_diff_std)

