from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_diabetes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import pandas as pd
from matplotlib import pyplot as plt

X, y = load_diabetes(return_X_y=True)

# Create the model
my_pipeline = Pipeline([("model", KNeighborsRegressor())])

my_model = GridSearchCV(estimator=my_pipeline, param_grid={
             'model__n_neighbors': [10, 15, 20, 25, 30]}, cv=5)

# Train the model with data
my_model.fit(X, y)
pd.DataFrame(my_model.cv_results_)

