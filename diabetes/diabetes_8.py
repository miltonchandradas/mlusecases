from sklearn.datasets import load_diabetes
from matplotlib import pyplot as plt
import pandas as pd

X, y = load_diabetes(return_X_y=True)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor

# Create the model
my_model = VotingRegressor([('clf1', RandomForestRegressor(n_estimators=100)), ('clf2', KNeighborsRegressor(n_neighbors=15)), ('clf3', LinearRegression())], weights=(1, 1, 2))

# Train the model with data
my_model.fit(X, y)

# Perform prediction
my_prediction = my_model.predict(X)

# Plot to see how well my prediction is doing
plt.scatter(my_prediction, y)

# Table to see how well my prediction is doing
pd.DataFrame({'Actual': y, 'Predict' : my_prediction})

