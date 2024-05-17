from sklearn.datasets import load_diabetes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import pandas as pd

X, y = load_diabetes(return_X_y=True, scaled=False)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor

# Create the model
my_pipeline = Pipeline([("scale", StandardScaler()), ("model", VotingRegressor([('clf1', RandomForestRegressor(n_estimators=100)), ('clf2', KNeighborsRegressor(n_neighbors=15)), ('clf3', LinearRegression())], weights=(1, 1, 2)))])

# Create the model
# my_model = VotingRegressor([('clf1', RandomForestRegressor(n_estimators=100)), ('clf2', KNeighborsRegressor(n_neighbors=15)), ('clf3', LinearRegression())], weights=(1, 1, 2))

# Train the model with data
my_pipeline.fit(X, y)

# Perform prediction
my_prediction = my_pipeline.predict(X)

# Plot to see how well my prediction is doing
plt.scatter(my_prediction, y)

# Table to see how well my prediction is doing
pd.DataFrame({'Actual': y, 'Predict' : my_prediction})

# save model
import pickle
MODEL_PATH = 'my_model.pkl'
pickle.dump(my_pipeline, open(MODEL_PATH, 'wb'))