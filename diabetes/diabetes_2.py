from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)

from sklearn.neighbors import KNeighborsRegressor

# Create the model
my_model = KNeighborsRegressor()

# Train the model with data
my_model.fit(X, y)

# Perform prediction
my_model.predict(X)
