from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.datasets import load_diabetes
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd
from matplotlib import pyplot as plt

X, y = load_diabetes(return_X_y=True)

LinearRegression().get_params()

model_params = {
    "random_regressor": {
        "model": RandomForestRegressor(random_state=10),
        "params": {
            "n_estimators": [70, 80, 90, 100, 110, 120]
        }
    },
    "linear_regressor": {
        "model": LinearRegression(),
        "params": {

        }
    },
    "kneighbors_regressor": {
        "model": KNeighborsRegressor(),
        "params": {
            "n_neighbors": [5, 10, 15, 20, 25, 30]
        }
    },
    "voting_regressor": {
        "model": VotingRegressor([('clf1', RandomForestRegressor(n_estimators=100)), ('clf2', KNeighborsRegressor(n_neighbors=15)), ('clf3', LinearRegression())]),
        "params": {
            "weights": [(1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 1, 1)]
        }
    },
}

scores = []

for key, model_type in model_params.items():
    classifier = GridSearchCV(
        model_type["model"], model_type["params"], cv=3, return_train_score=False)
    classifier.fit(X, y)

    scores.append({
        "model": key,
        "best_score": classifier.best_score_,
        "best_params": classifier.best_params_
    })

df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(df)
