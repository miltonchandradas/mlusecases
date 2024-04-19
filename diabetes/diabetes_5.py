from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_diabetes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import pandas as pd
from matplotlib import pyplot as plt

X, y = load_diabetes(return_X_y=True, scaled=False)

# Create the model
my_pipeline = Pipeline([("scale", StandardScaler()), ("model", KNeighborsRegressor())])

# Train the model with data
my_pipeline.fit(X, y)

# Perform prediction
my_prediction = my_pipeline.predict(X)

# Plot to see how well my prediction is doing
plt.scatter(my_prediction, y)

