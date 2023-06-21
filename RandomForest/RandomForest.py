import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Let's create an object of iris dataset
iris = load_iris()

# Create a dataframe from the object
df = pd.DataFrame(iris.data, columns = iris.feature_names)
# Let's check the dataset
print(df.head())