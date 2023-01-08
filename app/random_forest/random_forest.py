import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Local
from domain.data_dealing.data_split_test_train import DataSplitTestTrain, split_data_for_test

# explicação importante ~19:00 https://www.youtube.com/watch?v=RtA1rjhuavs
# sobre max_features e as influências. Posso discutir isso também.

# TODO os valores do loop tem que ser passados por DI, como configuração
# daí por um acaso  tenho valores padrão. Melhor.

def basic_random_forest_loop(X, y):
    
    # A conclusão a princípio é que depende muito mais da qtde e qualidade dos dados do que qualquer outra coisa
    for max_leaf_nodes in range(6, 8):
        for min_samples_leaf in range(1, 3):
            for i_test_size in range(1, 2):
                for i in range(0, 7):
                    # Parameters
                    test_size=0  + i_test_size/10
                    n_estimators = 50 + i*50
                    
                    # Model creation
                    model = RandomForestRegressor(random_state=0,
                                                    n_estimators=n_estimators,
                                                    min_samples_leaf=min_samples_leaf,
                                                    max_leaf_nodes=max_leaf_nodes)
                    
                    # Fitting
                    data = split_data_for_test(X, y, test_size=test_size)
                    model.fit(data.train.X, data.train.y)
                    
                    # Testing
                    predictions = model.predict(data.test.X)
                    acc_total = r2_score(predictions, data.test.y)
                    
                    # Debug & Printing
                    print('----------')
                    print(f'nº {i} | test_size={test_size} | n_estimators = {n_estimators} | min_samples_leaf = {min_samples_leaf} | max_leaf_nodes={max_leaf_nodes}')
                    print(f'acc = {acc_total}')
                    # Cuidado que esse aqui é EM VALORES ABSOLUTOS!
                    mean_error = mean_squared_error(predictions, data.test.y)
                    print(f'mean error² = {mean_error}')
                    pass