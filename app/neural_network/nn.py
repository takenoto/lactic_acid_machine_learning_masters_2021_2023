# TODO implementaçao scikit
# TODO implementação pytorch
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler, RobustScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline




def scikit_nn_loop(X, y, debug_print=False):

    # Sinceramente eu acho que esse do shuffle n tá prestando, ou então eu n entendi como
    # ele funciona. Pq a precisão tá alta demais.
    # X, x_test, y, y_test = train_test_split(X, y, test_size=0.92)
    

    param_grid = {
        'mlp_reg__solver': ['adam'],# 'lbfgs', 'sgd', ],
        # TODO como decidir quantas camadas, a qtde em cada uma, etc??
        # ref: https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
        'mlp_reg__hidden_layer_sizes': [(4, 4, 4, 3)],#(4, 3), (10, 5, 10, 5, 10, 5)], # (10,10)],#(40, 60)],# (10, 20), (60, 40), (200, 200)],
        'mlp_reg__random_state':[1],
        'mlp_reg__max_iter':[2000],
        #'shuffle':[True],
        'mlp_reg__verbose':[False],
        'mlp_reg__alpha':[0.0001],#, 1e-5]
        'mlp_reg__activation':['relu', 'tanh',],#'logistic']
        # 'mlp_reg__tol': [1e-3] #erro 10x mais alto que o padrão
    }


    reg = MLPRegressor()
    # Como os valores absolutos influenciam muito, acho que normalizar pode é atrapalhar
    # Pq considera mais proporções entre um e outro e os pontos min e máximos
    pipeline = Pipeline(
        [   
            # Esse scaler tava contribuindo MUITO com o overfitting e com
            # os dados que ficavam abaixo de 0 (P e S)
            # Com o standard scaler fica melhor, anote lá
            # ('standard_scaler', StandardScaler()), #Minmax fica bem ruim
            # ('robust_scaler', RobustScaler()),
            # ('poly', PolynomialFeatures(degree=3)), # Piorou mto nos meus testes
            ('mlp_reg', reg)
        ]
    )


    cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42) #test_size=0.8
    

    grid = GridSearchCV(pipeline,
                        param_grid,
                        error_score='raise',
                        cv=cv
                        )

    grid.fit(X, y) 

    print('\n---------------------------\n')
    print('grid: SCIKIT NEURAL NETWORK')
    print(grid.best_params_)
    print(grid.best_score_)
    print('\n---------------------------\n\n')


    return grid

    pass

def pytorch_nn_loop(X,y, debug_print=False):
    pass

def test():
    # ex from https://stackoverflow.com/questions/44548853/neural-network-with-multiple-outputs-in-sklearn
    model = MLPRegressor(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes = (15,10, 8, 9), #(50,15), #(5,2),
                        random_state=1)

    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8,9], [10, 11, 12] ])
    y = np.array([ [1, 2], [8,10], [14, 16], [20, 22] ])
    model.fit(X, y)
    print(f'score = {model.score(X, y)}')
    y_pred = model.predict(X)
    plt.plot(X[:,1], y[:,1], label='original')
    plt.plot(X[:,1], y_pred[:,1], label='predicted')
    plt.legend()
    plt.show()
    pass

if __name__ == '__main__':
    test();