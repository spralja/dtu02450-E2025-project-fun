#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model as lm
from sklearn import tree

#%% Load the dataset
# Importing the data
filename = 'data/glass+identification/glass.csv'

data = pd.read_csv(filename)


attributeNames = np.asarray(data.columns)[1:]

rawvalues = data.values
X = rawvalues[:, 1:-1] # Exclude first (Id) and last (Type) columns
y = rawvalues[:, -1] # Last column (Type)

N, M = X.shape

print(f'Number of observations: {N}')

C = 7
classNames = ['building_windows_float_processed', 'building_windows_non_float_processed', 'vehicle_windows_float_processed', 'vehicle_windows_non_float_processed', 'containers', 'tableware', 'headlamps']


# Remove one random row from the data to be used later for testing
random_row = np.random.randint(0, N)
X = np.delete(X, random_row, axis=0)
y = np.delete(y, random_row, axis=0)
N, M = X.shape

test_row = rawvalues[random_row, 1:-1]
test_label = rawvalues[random_row, -1]



# Try fitting a classification tree to the data



# Fit regression tree classifier, Gini split criterion, no pruning
criterion = "gini"
# dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=2)
dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=5.0 / N)
dtc = dtc.fit(X, y)

# convert the tree into a png file using the Graphviz toolset
fname = "tree_ex513_" + criterion + ".png"

# Visualize the graph (you can also inspect the generated image file in an external program)
# NOTE: depending on your setup you may need to decrease or increase the figsize and DPI setting
# to get a useful plot. Hint: Try to maximize the figure after it displays.

# fig = plt.figure(figsize=(4, 4), dpi=600)
# _ = tree.plot_tree(dtc, filled=False, feature_names=attributeNames)
# plt.savefig(fname)
# plt.show()



# Predict the label of the test row

x = test_row.reshape(1, -1)

x_class = dtc.predict(x)[0]
print(x_class)

# Print results
print("\nTest object attributes:")
print(dict(zip(attributeNames, x[0])))
print("\nClassification result:")
print(classNames[int(x_class)-1])
print("\nTrue label:")
print(classNames[int(test_label)-1])


#%%

class Classification_Model:
    def __init__(self,X,y,classNames):
        self.X = X
        self.y = y
        self.N, self.M = X.shape
        self.classNames = classNames
        self.C = len(classNames)
        self.standardized = False

    def standardize_data(self):
        self.mean = np.mean(self.X, 0)
        self.std = np.std(self.X, 0)
        self.X_hat = (self.X - np.ones((self.N, self.M)) * self.mean) / (np.ones((self.N, self.M)) * self.std)
        self.standardized = True

    def fit_classification_tree(self, criterion='gini', min_samples_split=5.0):
        try:
            if self.standardized:
                self.dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=min_samples_split / self.N)
                self.dtc = self.dtc.fit(self.X_hat, self.y)
            else:
                self.dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=min_samples_split / self.N)
                self.dtc = self.dtc.fit(self.X, self.y)
        except:
            print("Error in fitting the classification tree")

    def plot_tree(self):
        fig = plt.figure(figsize=(4, 4), dpi=600)
        _ = tree.plot_tree(dtc, filled=False, feature_names=attributeNames)
        plt.show()

    def predict(self, x):
        if self.standardized:
            x = (x - self.mean) / self.std
        x_class = self.dtc.predict(x)[0]
        return x_class


#%%

M1 = Classification_Model(X,y,classNames)
M1.fit_classification_tree()
M1.plot_tree()


#%%
