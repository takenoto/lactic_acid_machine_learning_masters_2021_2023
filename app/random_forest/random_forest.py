import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, ShuffleSplit

# Local
from domain.data_dealing.data_split_test_train import DataSplitTestTrain, split_data_for_test

# explicação importante ~19:00 https://www.youtube.com/watch?v=RtA1rjhuavs
# sobre max_features e as influências. Posso discutir isso também.

# TODO os valores do grid_params

def basic_random_forest_loop(X, y, debug_print=False):

    print('\n---------------------------\n')
    print('grid search: RANDOM TREE')
    param_grid = {
        'criterion':['absolute_error'],
        'random_state': [50],
        #'max_leaf_nodes':[3, 12],#5,6],
        #'min_samples_leaf':[4,20],#1, 2],
        #'min_samples_split': [4,20],
        'n_estimators':[50, 150],#400]
    }

    forest = RandomForestRegressor()
    cv = ShuffleSplit(n_splits=20, test_size=0.3, random_state=42)
    grid = GridSearchCV(forest, param_grid, error_score='raise', cv=cv)
    grid.fit(X, y)
    print('\n---------------------------\n\n')
    print(grid.best_params_)
    print(grid.best_score_)
    return grid