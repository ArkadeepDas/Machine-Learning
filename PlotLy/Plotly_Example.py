# Plot data in 3D-Plots using Plotly
# Importing libraries
import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
# Create dataframe from the dataset
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
# Let's check the data
print(df.head())

# Present columns
for column in df.columns:
    print(column)

# Let's create 3D scatter plot
fig = go.Figure(data=[
    go.Scatter3d(
        x=df['sepal length (cm)'],
        y=df['sepal width (cm)'],
        z=df['petal length (cm)'],
        marker=dict(
            size=5, color=df['target'], colorscale='Viridis', opacity=0.8))
])

# Set some layout
fig.update_layout(title='Iris Dataset - 3D Plot',
                  scene=dict(xaxis_title='Sepal Length (CM)',
                             yaxis_title='Sepal Width (CM)',
                             zaxis_title='Petal Length (CM)'))
fig.show()