#%%
# %pip install scikit-learn

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model as lm
from sklearn import model_selection

from dtuimldmtools import rlr_validate

#%%
# ANN packages
import importlib_resources
import torch
from scipy import stats
from scipy.io import loadmat

from dtuimldmtools import draw_neural_net, train_neural_net



#%% Load the dataset
# Importing the data
filename = 'data/glass+identification/glass.csv'

data = pd.read_csv(filename)

attributeNames = np.asarray(data.columns)[2:]

attributeNames = [format(name) for name in attributeNames]


#%%

rawvalues = data.values

# Set the X values to be the variables we are fitting from and the y values to be the refractive index
X = rawvalues[:, 2:-1] # Exclude Id, RI and Type columns
y = rawvalues[:, 1] # RI column

# Remove Na, Mg, Ba, and Fe columns as we know their correlations are insiginificant
# X = np.delete(X, [0, 1, -2, -1], axis=1) # Remove Na, Mg, Ba, and Fe columns


#%%
# Functions for the models

def r_linear_regression(X, y, lambdas, K):
    
    N, M = X.shape
    M = M + 1

    # Add the bias term to the input data
    X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)

    CV = model_selection.KFold(K, shuffle=True)

    # Initialize variables
    Error_train_rlr = np.empty((K, 1))
    Error_test_rlr = np.empty((K))
    w_rlr = np.empty((M, K))
    mu = np.empty((K, M - 1))
    sigma = np.empty((K, M - 1))

    k = 0

    best_err = 1e100
    
    # Analyze different values of lambda
    for s in range(len(lambdas)):
        
        lambda_s = lambdas[s]
        error_s = 0

        for train_index, test_index in CV.split(X, y):
            # extract training and test set for current CV fold
            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]
            
            # Standardize
            mu[k, :] = np.mean(X_train[:, 1:], 0)
            sigma[k, :] = np.std(X_train[:, 1:], 0)
            X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
            X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]


            # Dot the matrices for the linear regression
            Xty = X_train.T @ y_train
            XtX = X_train.T @ X_train

            # Estimate weights for the optimal value of lambda, on entire training set
            lambdaI = lambda_s * np.eye(M)
            lambdaI[0, 0] = 0  # Do no regularize the bias term
            w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
            
            # Compute mean squared error with regularization
            Error_train_rlr[k] = (
                np.square(y_train - X_train @ w_rlr[:, k]).sum(axis=0) / y_train.shape[0]
            )
            Error_test_rlr[k] = (
                np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]
            )
            error_s += Error_test_rlr[k]  # Add error for the current fold

        if error_s < best_err:
            # best_weights = w_rlr[:, k]
            best_err = error_s
            best_lambda = lambda_s
        
    mu = np.mean(X[:, 1:], 0)
    sigma = np.std(X[:, 1:], 0)

    # Standardize the entire dataset
    X[:, 1:] = (X[:, 1:] - mu) / sigma
    
    Xty = X.T @ y
    XtX = X.T @ X

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = best_lambda * np.eye(M)
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    best_weights = np.linalg.solve(XtX + lambdaI, Xty).squeeze()


    return (
        best_weights,
        best_lambda,  # optimal lambda
    )




def nn_regression(X, y, n_hidden_units_list, n_replicates, max_iter, K, comments=True):

    try:
        if y.shape != (X.shape[0], 1):
            y = y.reshape((X.shape[0], 1))
            # print("y reshaped to column vector")
            # print(f'y shape: {y.shape}')
    except:
        raise ValueError("y must be a column vector")

    CV = model_selection.KFold(K, shuffle=True)

    best_error = 1e100  # initialize best loss to a large number
    errors = []  # make a list for storing generalizaition error in each loop

    for n_hidden_units in n_hidden_units_list:
            
        # Define the model
        model = lambda: torch.nn.Sequential(
            torch.nn.Linear(M, n_hidden_units),  # M features to n_hidden_units
            torch.nn.Tanh(),  # 1st transfer function,
            torch.nn.Linear(n_hidden_units, 1),  # n_hidden_units to 1 output neuron
        )
        error_n = 0

        for k, (train_index, test_index) in enumerate(CV.split(X, y)):
        
        
    
            loss_fn = torch.nn.MSELoss()  # mean-squared-error loss 
                
            # Extract training and test set for current CV fold, convert to tensors
            X_train = torch.Tensor(X[train_index, :])
            y_train = torch.Tensor(y[train_index])
            X_test = torch.Tensor(X[test_index, :])
            y_test = torch.Tensor(y[test_index])

            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(
                model,
                loss_fn,
                X=X_train,
                y=y_train,
                n_replicates=n_replicates,
                max_iter=max_iter,
            )

            # Determine estimated class labels for test set
            y_test_est = net(X_test)

            # Determine errors and errors
            se = (y_test_est.float() - y_test.float()) ** 2  # squared error
            mse = (sum(se).type(torch.float) / len(y_test)).data.numpy()  # mean
            errors.append(mse)  # store error rate for current CV fold
            error_n += mse  # Add error for the current fold
        
        error_n /= K  # Average error for the current fold

        if error_n < best_error:
            best_error = error_n
            # print(f'new best loss: {best_loss}')
            best_net = net
            best_complexity = n_hidden_units
            # print(f'new best complexity: {best_complexity} with error: {best_error}')

    
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, n_hidden_units),  # M features to n_hidden_units
        torch.nn.Tanh(),  # 1st transfer function,
        torch.nn.Linear(n_hidden_units, 1),  # n_hidden_units to 1 output neuron
        )
    
    X = torch.Tensor(X)
    y = torch.Tensor(y)
    
    net, final_loss, learning_curve = train_neural_net(
        model,
        loss_fn,
        X=X,
        y=y,
        n_replicates=best_complexity,
        max_iter=max_iter,
        )

    



    return net, best_complexity, best_error



# #%%
# The two-level cross-validation

# Values of lambda
lambdas = np.power(10.0, range(-8, 6))
K = 10
CV = model_selection.KFold(K, shuffle=True)


N, M = X.shape

K_inner = 3
# Empty for baseline
Error_test_nofeatures = np.empty((K, 1))

# Values for rlr
rlr_weight_list = np.empty((K, M + 1))
Error_test_rlr = np.empty((K, 1))
found_lambdas = np.empty((K, 1))

# Values for nn
n_hidden_units = [1,2,3]  # number of hidden units
n_replicates = 1  # number of networks trained in each k-fold
max_iter = 10000
Error_test_nn = np.empty((K, 1))
h_list = np.empty((K, 1))


for k, (train_index, test_index) in enumerate(CV.split(X, y)):
    
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    

    # Standardize X_test_rlr to make it compatible with the weights from rlr
    mu = np.mean(X_test[:, :], 0)
    sigma = np.std(X_test[:, :], 0)
    X_test_hat = (X_test[:, :] - mu) / sigma
    X_test_rlr = np.concatenate((np.ones((X_test_hat.shape[0], 1)), X_test_hat), 1)


    # Error for baseline model (mean of training data)
    Error_test_nofeatures[k] = (
        np.square(y_test - y_test.mean()).sum(axis=0) / y_test.shape[0]
    )
    # print(f'Error_test_nofeatures: {Error_test_nofeatures[k]}')

    rlr_weights, rlr_lambda = r_linear_regression(X_train, y_train, lambdas, K_inner)    
    Error_test_rlr[k] = (
        np.square(y_test - X_test_rlr @ rlr_weights).sum(axis=0) / y_test.shape[0]
    )
    
    rlr_weight_list[k, :] = rlr_weights # store weights for current CV fold
    found_lambdas[k] = rlr_lambda # store lambda for current CV fold

    # NN part
    net, h_list[k], nn_best_error = nn_regression(X_train, y_train, n_hidden_units, n_replicates, max_iter, K_inner, comments=True)
    
    # Make tensor variants for the test set
    X_test_nn = torch.Tensor(X_test)
    y_test_nn = torch.Tensor(y_test.reshape((X_test.shape[0], 1)))

    
    # Determine estimated class labels for test set
    y_test_est = net(X_test_nn)

    # Determine errors and errors
    se = (y_test_est.float() - y_test_nn.float()) ** 2  # squared error
    mse = (sum(se).type(torch.float) / len(y_test_nn)).data.numpy()  # mean
    Error_test_nn[k] = mse  # store error rate for current CV fold
    



    

#%%
rlr_attribute_names = np.concatenate(
    (["bias"], attributeNames[:-1]), 0
)  # Add bias term to the names

# Remove Na, Mg, Ba, and Fe columns as we know their correlations are insiginificant
# rlr_attribute_names = np.delete(rlr_attribute_names, [1, 2, -2, -1], axis=0) # Remove Na, Mg, Ba, and Fe columns

# Baseline Summary
print(f'Found errors for baseline:\n {Error_test_nofeatures}')

# Rlr Summary
print(f'Found errors for rlr:\n {Error_test_rlr}')
print(f'Weights for rlr:\n {rlr_attribute_names} \n {rlr_weight_list}')
print(f'Largest weight for rlr: {np.max(rlr_weight_list[:,1:]):.5f} for {rlr_attribute_names[1 +np.argmax(np.mean(rlr_weight_list[:,1:], axis=0))]}')

# nn Summary
print(f'Found errors for nn:\n {Error_test_nn}')
print(f'Complexities for nn: {h_list}')

avg_nn_error = np.mean(Error_test_nn)
avg_rlr_error = np.mean(Error_test_rlr)
avg_baseline_error = np.mean(Error_test_nofeatures)

min_nn_error = np.min(Error_test_nn)
min_rlr_error = np.min(Error_test_rlr)
min_baseline_error = np.min(Error_test_nofeatures)
#%%
print('shapes: h_list, Error_test_nn, found_lambdas, Error_test_rlr, Error_test_nofeatures')
print(f'{h_list.shape}, {Error_test_nn.shape}, {found_lambdas.shape}, {Error_test_rlr.shape}, {Error_test_nofeatures.shape}')

#%%
# Print a table of the results
print("\n\n")
print('Results for the different models')
print('Outer fold    ANN       Linear regression    Baseline')
print('i          h_i   E_i       lambda_i    E_i         E_i')
for i in range(K):
    print(f'{i+1}          {h_list[i,0]}   {Error_test_nn[i,0]}   {found_lambdas[i,0]}    {Error_test_rlr[i,0]}    {Error_test_nofeatures[i,0]}')


#%%
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

#%%
# Print a table of the results
print("\n\n")

print("  Results for the different models")
print("  -------------------------------")
print("  Model                  Test error average")
print(f'Baseline model: {avg_baseline_error:.8f}')
print(f'Linear regression: {avg_rlr_error:.8f}')
print(f'Neural network: {avg_nn_error:.8f}')

print("  -------------------------------")
print("  Model                  Test error min")
print(f'Baseline model: {min_baseline_error:.8f}')
print(f'Linear regression: {min_rlr_error:.8f}')
print(f'Neural network: {min_nn_error:.8f}')
print(f'Complexity for best nn: {h_list[np.argmin(Error_test_nn)]}')

