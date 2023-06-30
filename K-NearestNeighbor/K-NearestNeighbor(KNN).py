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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

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

# Create the K-NN Classifier
# Here we only set number of neighbors = 5. We can set multiple parameters
# metric: 'minkowsik'. We can use other values there
knn = KNeighborsClassifier(n_neighbors=7)
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Training >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
knn.fit(X_train, Y_train)
print('Complete Training')

# Now let's check the accuracy
# We get 100% accuracy in 3 and 5.
print(knn.score(X_test, Y_test))

# Let's see the confusion matrix
Y_pred = knn.predict(X_test)
c_matrix = confusion_matrix(Y_test, Y_pred)
print(c_matrix)

# Let's check tyhe classification report
print(classification_report(Y_test, Y_pred))