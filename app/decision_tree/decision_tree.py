import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
# Metrics
# from sklearn.metrics import accuracy_score # esse é para coisas do tipo classificação
from sklearn.metrics import r2_score

from sklearn.ensemble import AdaBoostRegressor

# Plotting
import matplotlib.pyplot as plt


#---------------- LOCAL

from domain.data_dealing.data_split_test_train import DataSplitTestTrain, split_data_for_test

def classic_decision_tree_loop(X, y, test_sizes=[0.1, 0.2]):
    model = None;
    for test_size in test_sizes:
        data = split_data_for_test(X, y, test_size=test_size)
        # Testa depth de 1 a 6
        for i in range(2, 5):
            max_depth = i;
            model = do_decision_tree(data.train.X, data.train.y, max_depth=max_depth)

            #-----------------------
            predictions = model.predict(data.test.X)

            # acc_x = r2_score(predictions[0], data.test.y[0])
            # print(f'acc x = {acc_x}')
            # acc_lla = r2_score(predictions[1], data.test.y[1])
            # print(f'acc lla = {acc_lla}')
            # acc_s = r2_score(predictions[2], data.test.y[2])
            # print(f'acc_s = {acc_s}')

            acc_total = r2_score(predictions, data.test.y)
            print(f'test_size = {test_size}')
            print(f'max_depth = {max_depth}')
            print(f'acc total = {acc_total}')

            # acc = check_model_accurracy();
            # TODO calcular acerto %
            # TODO na verdade acho que cada modelo tem que ser um objeto
            # e ter os indicadores de acertos internamente??? Se variar né...

            #-------------------------
            # pega predição do modelo
            # Esses são os dados do experimento 3
            if(False):

                fig3datafor_plot = np.array([
                    [1.15, 3.5, 36],
                    [1.5, 5, 34],
                    [2, 5.8, 34],
                    [2.4, 6, 31],
                    [2.95 , 7, 25.3],
                    [3.6, 9, 21],
                    [4.5, 13, 20],
                    [5.7, 16.8, 17.3],
                    [6.3, 23, 12.5],
                    [6.85, 24, 6.8],
                    [6.9, 28, 1],
                    [6.93, 30, 0.1]
                    ])

                row_for_fig_3 = np.array([
                        [0, 1.15, 36, 3.5],
                        [1, 1.15, 36, 3.5],
                        [2, 1.15, 36, 3.5],
                        [3, 1.15, 36, 3.5],
                        [4, 1.15, 36, 3.5],
                        [5, 1.15, 36, 3.5],
                        [6, 1.15, 36, 3.5],
                        [7, 1.15, 36, 3.5],
                        [8, 1.15, 36, 3.5],
                        [9, 1.15, 36, 3.5],
                        [10, 1.15, 36, 3.5],
                        [11, 1.15, 36, 3.5],
                        ])
                yhat = model.predict(row_for_fig_3)
                plt.plot(row_for_fig_3[:, 0], fig3datafor_plot[:,0], label='X_real')
                plt.plot(row_for_fig_3[:, 0], fig3datafor_plot[:,1], label='P_real')
                plt.plot(row_for_fig_3[:, 0], fig3datafor_plot[:,2], label='S_real')
                plt.plot(row_for_fig_3[:, 0], yhat[:, 0], label='X')
                plt.plot(row_for_fig_3[:, 0], yhat[:, 1], label='P')
                plt.plot(row_for_fig_3[:, 0], yhat[:, 2], label='S')
                # plt.plot(yhat[:, 0], yhat[:, 1], label='first')
                plt.legend()
                plt.show()

    return model;


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



    


def do_decision_tree(X, y, max_depth=3):
    model = DecisionTreeRegressor(max_depth=max_depth)
    # model = Pipeline(
    #     [
    #         ('scale', StandardScaler()),
    #         ('model', DecisionTreeRegressor(max_depth=max_depth))
    #     ]
    # )
    model.fit(X, y)

    return model
