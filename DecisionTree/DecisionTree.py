import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
from sklearn import tree
from matplotlib import pyplot as plt

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
# 'criterion': 'gini', # Function measure the quality of split.
# 'max_depth': None, # How deep can tree be(Try different value and check which one is the best)
# 'max_features': None, 
# 'max_leaf_nodes': None, 
# 'min_impurity_decrease': 0.0, 
# 'min_samples_leaf': 1, 
# 'min_samples_split': 2, # Minimum number of sample require to split internal node.
# 'min_weight_fraction_leaf': 0.0, 
# 'random_state': None, 
# 'splitter': 'best'}

# We can change these parameters and experiment
#############################################################################

# Predict the model
predictions = classification.predict(X_test)
print(predictions)

# Decision tree can also gives us the probability output
probability_predictions = classification.predict_proba(X_test)
print(probability_predictions)
# The output structure is [class0, class1]
# The tree split is prefect. So each and every class predict either 0 or 1. Otherwise maximum vote wins.
# If we change the 'max_depth = 4' then we can see the change of the output probability prediction.

# Let's check the output score
accuracy = accuracy_score(Y_test, predictions)
print(accuracy) 

# Let's check the confusion matrix
conf_mtx = confusion_matrix(Y_test, predictions)
print(conf_mtx)

# Let's check the precision and recall
precision = precision_score(Y_test, predictions)
recall = recall_score(Y_test, predictions)
print('Precision: ', precision)
print('Recall: ', recall)

# Classification report is a very good function which shows everything together
print(classification_report(Y_test, predictions, target_names = ['Malignant', 'Benign']))

# Let's see the feature importance of Decision tree
feature_importance = pd.DataFrame(classification.feature_importances_, index = X_train.columns).sort_values(0, ascending = False)
print(feature_importance)

# We can plot the tree to see how the Decision Tree works
plt.figure(figsize = (32, 32))
tree.plot_tree(classification, feature_names = X_train.columns,
                   class_names = {0: 'Malignant', 1: 'Behign'},
                   filled = True,
                   fontsize = 14)
plt.savefig('DecisionTree.png')