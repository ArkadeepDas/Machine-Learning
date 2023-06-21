import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Let's create an object of iris dataset
dataset = load_iris()
# iris here is a python dictionary
# 'feature_names' holds the name of the features
# 'target' holds the values 0 and 1
# 'data' holds the values of each columns

# Create a dataframe from the object
X = pd.DataFrame(dataset.data, columns = dataset.feature_names)
# Let's check the dataset
print(X.head())

# Let's add the target column
Y = pd.DataFrame(dataset.target)
print(Y.head())

# Let's split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .3)
# Check the length of the data
print('Train data length: ', len(X_train))
print('Test data length: ', len(X_test))