import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score

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

# Create a Random Forest Classifier
random_forest = RandomForestClassifier(n_jobs = 1, random_state = 1)
#############################################################################
# Parameters Used in Random Forest

# We can play with many variables
# n_jobs = 1: Priorities the job, how to run accross the system
# max_features = 'auto': Automatically take features from dataset. If we have more features then we can set some values
# max_leaf_nodes = None: Ultimate number of leaf nodes
# criteron = 'gini': Function to measure the quality of a split.
# max_depth = None: Maximum depth of the tree

# We can change many parameters and experiment
#############################################################################

# Let's train the model
random_forest.fit(X_train, Y_train)

# Let's predict the model
predictions = random_forest.predict(X_test)
print(predictions)

# We can view the predicted probabilities of the features
prediction_probabilities = random_forest.predict_proba(X_test)
print(prediction_probabilities)

# Now let's map names for the plants for each predictions
prediction_names = dataset.target_names[random_forest.predict(X_test)]
print(prediction_names[:5])

# Let's check the confusion matrix
conf_mtx = confusion_matrix(Y_test, predictions)
print(conf_mtx)