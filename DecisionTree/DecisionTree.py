import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Let's load the dataset
data = load_breast_cancer()
# Convert dataset to pandas dataframe
X = pd.DataFrame(data = data['data'], columns = data['feature_names'])
Y = pd.DataFrame(data.target)

# Let's Split the dataset
# We don't do any data normalization here
# In general decision don't require any data normalization/scaling
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33)

# It's a classification problem
decission_tree_classifier = DecisionTreeClassifier()
classification = decission_tree_classifier.fit(X_train, Y_train)

# Check the parameters that are used
print(classification.get_params())

#############################################################################
# Parameters Used in Decision Tree

# {'ccp_alpha': 0.0, 
# 'class_weight': None, 
# 'criterion': 'gini', 
# 'max_depth': None, 
# 'max_features': None, 
# 'max_leaf_nodes': None, 
# 'min_impurity_decrease': 0.0, 
# 'min_samples_leaf': 1, 
# 'min_samples_split': 2, 
# 'min_weight_fraction_leaf': 0.0, 
# 'random_state': None, 
# 'splitter': 'best'}

# We can change these parameters and experiment
#############################################################################

