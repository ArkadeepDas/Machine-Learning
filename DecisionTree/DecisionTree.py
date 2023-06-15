import pandas as pd
from sklearn.datasets import load_breast_cancer

# Let's load the dataset
dataset = load_breast_cancer()
# Convert dataset to pandas dataframe
dataset = pd.DataFrame(data = dataset['data'], columns = dataset['feature_names'])

# Let's check the data
print(dataset.head())