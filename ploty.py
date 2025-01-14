import plotly.express as px
import pandas as pd

# Load the Iris dataset from seaborn and convert it to a DataFrame
df = px.data.iris()

# Create an interactive 3D scatter plot
fig = px.scatter_3d(df, x='sepal_width', y='sepal_length', z='petal_length', color='species',
                    title='3D Scatter Plot of Iris Dataset',
                    labels={'sepal_width': 'Sepal Width', 'sepal_length': 'Sepal Length', 'petal_length': 'Petal Length'})

# Show the plot
fig.show()
