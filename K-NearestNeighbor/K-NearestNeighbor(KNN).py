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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = load_iris()
# Checking feature names
print(data.feature_names)
# Check target names
print(data.target_names)

# Let's load the data in a dataframe
df = pd.DataFrame(data.data, columns=data.feature_names)
# Adding target column
df['target'] = data.target
# Check the data
print(df.head())

# Here first 50 are Setosa
# From 50 - 100 are Versicolor
# From 100 - 150 are Virginica

# So let's split the data and plot them
setosa = df[:50]
versicolor = df[50:100]
verginica = df[100:]

# Now let's try to plot them
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(setosa['sepal length (cm)'],
            setosa['sepal width (cm)'],
            color='green',
            marker='+')
plt.scatter(versicolor['sepal length (cm)'],
            versicolor['sepal width (cm)'],
            color='blue',
            marker='.')
plt.scatter(verginica['sepal length (cm)'],
            verginica['sepal width (cm)'],
            color='red',
            marker='*')
plt.savefig('ClusterStructure.png')

# Let's split the data set
X = df.drop(['target'], axis='columns')
print(X)
Y = df['target']
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=1)
# Let's Check the length of the data
print('Length of training data: ', len(X_train))
print('Length of the testing data: ', len(X_test))