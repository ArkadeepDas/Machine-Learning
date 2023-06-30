# K-Nearest Neighbor is basic yet essential classification algorithm in Machine Learning
# It belongs to supervised learning
# For KNN we need to calculate distance matrix
# We have: 1) Euclidean Distance 2) Manhattan Distance 3) Minkowsik Distance

# Basically we plot all the data points and for a upcoming data we calculate the distance and tell which class it belongs too
# We have to figureout the value of K
# We have to figureout nearby most K data points using any above distance matrix and magority vote wins
# K should not be very high on not very low(depends on data)

# Here we use Iris flower dataset

import pandas as pd
from sklearn.datasets import load_iris

data = load_iris()
# Checking feature names
print(data.feature_names)
# Check target names
print(data.target_names)

# Let's load the data in a dataframe
df = pd.DataFrame(data.data, columns = data.feature_names)
# Adding target column
df['target'] = data.target
# Check the data
print(df.head())