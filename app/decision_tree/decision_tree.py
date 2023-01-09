import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, ShuffleSplit
# from sklearn.metrics import accuracy_score # esse é para coisas do tipo classificação
from sklearn.metrics import r2_score

from sklearn.ensemble import AdaBoostRegressor

# Plotting
import matplotlib.pyplot as plt


#---------------- LOCAL

from domain.data_dealing.data_split_test_train import DataSplitTestTrain, split_data_for_test

def classic_decision_tree_loop(X, y, test_sizes=[0.1, 0.2]):
    model = None

    param_grid = {
        'random_state': [50],
        #'criterion':['squared_error', 'absolute_error'],
        'criterion':['absolute_error'],
        'max_depth':[2, 10, 20],#,50],
        # Com mais leafs fica uma escadinha mas faz mais sentido pra n dar overfit
        'min_samples_leaf': [4, 12] #[1, 2, 10]
    }

    tree = DecisionTreeRegressor()
    cv = ShuffleSplit(n_splits=20, test_size=0.8, random_state=42)
    
    grid = GridSearchCV(tree,
                        param_grid, error_score='raise',
                        # TODO o que é esse scoring?
                        #scoring='f1',
                        cv=cv, # nenhum dos dois cvs consegui fazer prestar
                        # verbose=3
                        )
    
    for test_size in [0.1]:
        grid.fit(X, y)

    # pd.DataFrame(grid.cv_results_)
    print('\n---------------------------\n')
    print('grid: decision tree')
    # TODO bom mesmo seria printar todos os erros disponíveis e pegar um razoável, pra n dar overfit
    print(grid.best_params_)
    print(grid.best_score_)
    print('\n---------------------------\n\n')

    return grid


def adaboost_decision_tree_loop(X, y, debug_print=False):
    """
    ref: https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_regression.html#id2
    """
    rng = np.random.RandomState(3)
    
    model = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=rng
    )

    model = AdaBoostRegressor(random_state=0)
    

    print('ADA BOOST + DECISION TREE')
    # Como y é uma lista de lista e ele só pega 1, tem que resolver de outra forma
    # daí separa 1 modelo pra cada feature
    print(f'leny0 = {len(y[0])}')
    for y_i in range(len(y[0])):
        for t in range(10, 40, 10):
            test_size = t/100
            
            data = split_data_for_test(X, y, test_size=test_size)
            
            model = AdaBoostRegressor(
                    DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=rng
                    )
           
            model.fit(data.train.X, data.train.y[:,y_i])
            predictions = model.predict(data.test.X)
            acc_total = r2_score(data.test.y[:,y_i], predictions)
            # Debug & PrintingQ 
            if(debug_print):      
                print(f'y_i = {y_i}')
                print(f'acc total = {acc_total}')

    return model