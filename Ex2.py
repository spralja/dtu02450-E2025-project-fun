#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import svd

#%%
# Importing the data
filename = 'data/glass+identification/glass.csv'

data = pd.read_csv(filename)
#%%

attributeNames = np.asarray(data.columns)[1:]

rawvalues = data.values
X = rawvalues[:, 1:-1]
y = rawvalues[:, -1]

N, M = X.shape

if X.any() < 0:
    print('Negative values in X')
else:
    print('No negative values in X')
if X.all() >= 0:
    print('All values in X are positive')


median = np.nanmedian(X, axis=0)
std = np.nanstd(X, axis=0)


print(median)
print(np.nanmean(y))
print(f'list of std {std}')

nan_count = np.sum(np.isnan(X), axis=0)
print(f'Nan count: {nan_count}')
print(f'Nan percentage: {nan_count/len(X[:,0])*100}%')
nan_total = np.sum(np.isnan(X))
print(f'Nan')


sum_temp = np.sum(X, axis=1)
nan_row_sum = np.sum(np.isnan(sum_temp))
print(f'Nan row sum: {nan_row_sum}')
print(f'Nan row percentage: {round(nan_row_sum/len(X[:,0])*100,2)}%')

#%%
#Dataset without NaN values
is_not_nan = np.isnan(X) < 1
# X_no_nan = X[~np.isnan(X).any(axis=1)]
X_no_nan = X[is_not_nan.all(axis=1)]


#%%
# Copied from excercise 2.3.2
# X_C = X
# X = X_no_nan
plt.figure(figsize=(8, 7))
u = np.floor(np.sqrt(M))
v = np.ceil(float(M) / u)
for i in range(M):
    plt.subplot(int(u), int(v), i + 1)
    plt.hist(X[:, i], color=(0.2, 0.8 - i * 0.2*(4/M), 0.4))
    plt.xlabel(attributeNames[i])
    plt.ylim(0, N / 2)
    plt.tight_layout()

plt.show()
# X = X_C

#%%

plt.figure()
plt.boxplot(X_no_nan)
plt.xticks(range(1, 10), attributeNames[:-1], rotation=45)
plt.title("Boxplot of data without NaN values")
plt.show()


#%%

# Looking at a standardized version of the data through boxplots

X_hat = [(X[:, i][~np.isnan(X[:, i])] - median[i])/ std[i] for i in range(9)]

plt.figure()
plt.boxplot(X_hat)
plt.xticks(range(1, 10), attributeNames[:-1], rotation=45)
plt.title("Boxplot of standardized data")
plt.show()


#%%
# Similarities

Cor_mat = np.empty((M,M))
for i in range(M):
    for j in range(M):
        Cor_mat[i,j] = np.corrcoef(X_no_nan[:,i], X_no_nan[:,j])[0,1]
Cor_sign = np.sign(Cor_mat)
# print(Cor_mat)
plt.figure()
plt.pcolormesh(np.abs(Cor_mat))
plt.colorbar()
plt.show()
print(Cor_mat[1,4])
print(Cor_mat[2,4])
print(np.max(np.abs(Cor_mat[Cor_mat < 0.9])))


#%%
C = 7
classNames = ['building_windows_float_processed', 'building_windows_non_float_processed', 'vehicle_windows_float_processed', 'vehicle_windows_non_float_processed', 'containers', 'tableware', 'headlamps']
# X_temp = X
# y_temp = y
# rand_choices = np.random.choice(len(y), int(len(y)/25), replace=False)

# X = X[rand_choices]
# y = y[rand_choices]

plt.figure(figsize=(12, 10))
for m1 in range(M):
    for m2 in range(M):
        plt.subplot(M, M, m1 * M + m2 + 1)
        for c in range(C):
            class_mask = y == c
            plt.plot(np.array(X[class_mask, m2]), np.array(X[class_mask, m1]), ".")
            if m1 == M - 1:
                plt.xlabel(attributeNames[m2])
            else:
                plt.xticks([])
            if m2 == 0:
                plt.ylabel(attributeNames[m1])
            else:
                plt.yticks([])
                            
plt.legend(classNames)

plt.show()

# X = X_temp
# y = y_temp



