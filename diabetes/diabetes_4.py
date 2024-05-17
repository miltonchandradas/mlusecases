from sklearn.datasets import load_diabetes
from matplotlib import pyplot as plt
import pandas as pd

X, y = load_diabetes(return_X_y=True)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

# Create the model
# my_model = KNeighborsRegressor(n_neighbors=1)
my_model = LinearRegression()
# my_model = KNeighborsRegressor()

# Train the model with data
my_model.fit(X, y)

# Perform prediction
my_prediction = my_model.predict(X)

# Plot to see how well my prediction is doing
plt.scatter(my_prediction, y)

# Table to see how well my prediction is doing
pd.DataFrame({'Actual': y, 'Predict' : my_prediction})

