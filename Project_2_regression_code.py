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
print(type(attributeNames))
#%%
# attributeNames = 

rawvalues = data.values
# X = rawvalues[:, 1:-1] # Exclude first (Id) and last (Type) columns
# y = rawvalues[:, -1] # Last column (Type)

# Set the X values to be the variables we are fitting from and the y values to be the refractive index
X = rawvalues[:, 2:-1] # Exclude Id, RI and Type columns
y = rawvalues[:, 1] # RI column



N, M = X.shape
M = M + 1

X_ANN = X # Save the original X for ANN later

X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
print(X.shape)

print(f'Number of observations: {N}')

C = 7
classNames = ['building_windows_float_processed', 'building_windows_non_float_processed', 'vehicle_windows_float_processed', 'vehicle_windows_non_float_processed', 'containers', 'tableware', 'headlamps']




#%%
# Functions for the models

def r_linear_regression(X, y, lambdas, K, comments=True):
    N, M = X.shape
    M = M + 1

    # Add the bias term to the input data
    X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)

    CV = model_selection.KFold(K, shuffle=True)

    # Initialize variables
    # T = len(lambdas)
    Error_train = np.empty((K, 1))
    Error_test = np.empty((K, 1))
    Error_train_rlr = np.empty((K, 1))
    Error_test_rlr = np.empty((K, 1))
    w_rlr = np.empty((M, K))
    mu = np.empty((K, M - 1))
    sigma = np.empty((K, M - 1))
    w_noreg = np.empty((M, K))

    k = 0
    lams = np.empty((K, 1))
    
    for train_index, test_index in CV.split(X, y):
        # extract training and test set for current CV fold
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        internal_cross_validation = 10
        
        (
            opt_val_err,
            opt_lambda,
            mean_w_vs_lambda,
            train_err_vs_lambda,
            test_err_vs_lambda,
        ) = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

        # Standardize
        mu[k, :] = np.mean(X_train[:, 1:], 0)
        sigma[k, :] = np.std(X_train[:, 1:], 0)

        # if sigma[k, :] != 0:
        X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
        X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]
        # else:
        #     X_train[:, 1:] = X_train[:, 1:] - mu[k, :]
        #     X_test[:, 1:] = X_test[:, 1:] - mu[k, :]

        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train

        # Estimate weights for the optimal value of lambda, on entire training set
        lambdaI = opt_lambda * np.eye(M)
        lambdaI[0, 0] = 0  # Do no regularize the bias term
        w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()

        # Compute mean squared error with regularization
        Error_train_rlr[k] = (
            np.square(y_train - X_train @ w_rlr[:, k]).sum(axis=0) / y_train.shape[0]
        )
        Error_test_rlr[k] = (
            np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]
        )

        lams[k] = opt_lambda

        # Estimate weights for unregularized linear regression, on entire training set
        w_noreg[:, k] = np.linalg.solve(XtX, Xty).squeeze()
        # Compute mean squared error without regularization
        Error_train[k] = (
            np.square(y_train - X_train @ w_noreg[:, k]).sum(axis=0) / y_train.shape[0]
        )
        Error_test[k] = (
            np.square(y_test - X_test @ w_noreg[:, k]).sum(axis=0) / y_test.shape[0]
        )

    best_val_err = np.min(np.mean(Error_test_rlr, axis=0))
    best_lambda = lambdas[np.argmin(np.mean(Error_test_rlr, axis=0))]
    best_weights = w_rlr[:, np.argmin(np.mean(Error_test_rlr, axis=0))]

    return (
        best_weights,
        best_lambda,  # optimal lambda
    )

def linear_regression_nofeature(X, y, K, comments=True):
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = (
        np.square(y_train - y_train.mean()).sum(axis=0) / y_train.shape[0]
    )
    Error_test_nofeatures[k] = (
        np.square(y_test - y_test.mean()).sum(axis=0) / y_test.shape[0]
    )



def nn_regression(X, y, n_hidden_units, n_replicates, max_iter, K, comments=True):

    try:
        if y.shape != (X.shape[0], 1):
            y = y.reshape((X.shape[0], 1))
            print("y reshaped to column vector")
            print(f'y shape: {y.shape}')
    except:
        raise ValueError("y must be a column vector")

    CV = model_selection.KFold(K, shuffle=True)

    # Define the model
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, n_hidden_units),  # M features to n_hidden_units
        torch.nn.Tanh(),  # 1st transfer function,
        torch.nn.Linear(n_hidden_units, 1),  # n_hidden_units to 1 output neuron
        # no final tranfer function, i.e. "linear output"
    )

    loss_fn = torch.nn.MSELoss()  # mean-squared-error loss
    print("Training model of type:\n\n{}\n".format(str(model()))) if comments else None
    errors = []  # make a list for storing generalizaition error in each loop
    for k, (train_index, test_index) in enumerate(CV.split(X, y)):
        print("\nCrossvalidation fold: {0}/{1}".format(k + 1, K)) if comments else None
            
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

        print("\n\tBest loss: {}\n".format(final_loss)) if comments else None

        # Determine estimated class labels for test set
        y_test_est = net(X_test)

        # Determine errors and errors
        se = (y_test_est.float() - y_test.float()) ** 2  # squared error
        mse = (sum(se).type(torch.float) / len(y_test)).data.numpy()  # mean
        errors.append(mse)  # store error rate for current CV fold
    
    # Print the average classification error rate
    if comments:
        print(
            "\nEstimated generalization error, RMSE: {0}".format(
                round(np.sqrt(np.mean(errors)), 4)
            )
        )

    return net, errors





#%%
# The two-level cross-validation

# Values of lambda
lambdas = np.power(10.0, range(-8, 6))
K = 10
CV = model_selection.KFold(K, shuffle=True)

N, M = X_ANN.shape

K_inner = 3
# Empty for baseline
Error_test_nofeatures = np.empty((K, 1))

# Values for rlr
rlr_weight_list = np.empty((K, M + 1))
Error_test_rlr = np.empty((K, 1))

# Values for nn
n_hidden_units = 2  # number of hidden units
n_replicates = 1  # number of networks trained in each k-fold
max_iter = 10000
Error_test_nn = np.empty((K, 1))


for k, (train_index, test_index) in enumerate(CV.split(X_ANN, y)):
    
    X_train = X_ANN[train_index]
    y_train = y[train_index]
    X_test = X_ANN[test_index]
    y_test = y[test_index]
    X_test_rlr = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), 1)

    # Error for baseline model (mean of training data)
    Error_test_nofeatures[k] = (
        np.square(y_test - y_train.mean()).sum(axis=0) / y_test.shape[0]
    )

    print(f'X_train shape: {X_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    # Regularized linear regression part
    rlr_weights, rlr_lambda = r_linear_regression(X_train, y_train, lambdas, K_inner, comments=True)    
    Error_test_rlr[k] = (
        np.square(y_test - X_test_rlr @ rlr_weights).sum(axis=0) / y_test.shape[0]
    )
    print(f'rlr_weights shape: {rlr_weights.shape}')
    rlr_weight_list[k, :] = rlr_weights

    # NN part
    net, errors = nn_regression(X_train, y_train, n_hidden_units, n_replicates, max_iter, K_inner, comments=True)

    # Make tensor variants for the test set
    X_test_nn = torch.Tensor(X_test)
    y_test_nn = torch.Tensor(y_test.reshape((X_test.shape[0], 1)))

    print(f'X_train shape: {X_train.shape}')
    print(f'X_test_nn shape: {X_test_nn.shape}')

    # Determine estimated class labels for test set
    y_test_est = net(X_test_nn)

    # Determine errors and errors
    se = (y_test_est.float() - y_test_nn.float()) ** 2  # squared error
    mse = (sum(se).type(torch.float) / len(y_test_nn)).data.numpy()  # mean
    Error_test_nn[k] = mse  # store error rate for current CV fold
    



    # print(rlr_lambda)
    

#%%
rlr_attribute_names = np.concatenate(
    (["bias"], attributeNames[:-1]), 0
)  # Add bias term to the names

# Baseline Summary
print(f'Found errors for baseline:\n {Error_test_nofeatures}')

# Rlr Summary
print(f'Found errors for rlr:\n {Error_test_rlr}')
print(f'Weights for rlr:\n {rlr_attribute_names} \n {rlr_weight_list}')
print(f'Largest weight for rlr: {np.max(rlr_weight_list[:,1:]):.5f} for {rlr_attribute_names[1 +np.argmax(np.mean(rlr_weight_list[:,1:], axis=0))]}')

# nn Summary
print(f'Found errors for nn:\n {np.sqrt(Error_test_nn)}')

avg_nn_error = np.mean(Error_test_nn)
avg_rlr_error = np.mean(Error_test_rlr)
avg_baseline_error = np.mean(Error_test_nofeatures)

min_nn_error = np.min(Error_test_nn)
min_rlr_error = np.min(Error_test_rlr)
min_baseline_error = np.min(Error_test_nofeatures)




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