# Foreign imports
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from files.files import load_lactic_acid_production_data;
from domain.data_dict_simple_label_remover import simple_label_remover


# Coisas para o loop local
from app.decision_tree.decision_tree import classic_decision_tree_loop, adaboost_decision_tree_loop
from app.k_neig.k_neg import basic_k_neig_loop
from app.random_forest.random_forest import basic_random_forest_loop
from app.neural_network.nn import scikit_nn_loop, pytorch_nn_loop

# TODO faz regressão decision_tree facílima
# só pra testar pegar esses dados em 3 pontos e plotar o gráfico,
# veja se dá pra trabalhar tranquilo nessa nomenclatura ou vai dar bode


# TODO os prefixos de entrada e saída
# os identificadores das substâncias
# isso tudo deve ficar salvo numa classe própria

# TODO # as features e targets dependem do modelo a ser analisado, daí não ficam nessa
# classe geral


# TODO faz um arquivo chamado exemplo que ensina passo a passo
# pra quando eu precisar consultar

def plotvalues(X, y, prediction, time_index=0, X_index=0, P_index=1, S_index=2, show_X=False, show_P=False, show_S=False):
    if(show_X):
        plt.plot(X[:, time_index], y[:,X_index], label='X_real')
        plt.plot(X[:, time_index], prediction[:, X_index], label='X')
    if(show_S):
        plt.plot(X[:, time_index], y[:,S_index], label='S_real')
        plt.plot(X[:, time_index], prediction[:, S_index], label='S')
    if(show_P):
        plt.plot(X[:, time_index], y[:,P_index], label='P_real')
        plt.plot(X[:, time_index], prediction[:, P_index], label='P')
    plt.legend()
    plt.show();
    
# TODO faz a iteração nesses moldes: https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_training_curves.html#sphx-glr-auto-examples-neural-networks-plot-mlp-training-curves-py
# Daí só seto os params e o resto vem de brinde

def main(run_decision_trees=False, run_neigs=False, run_gradient_boosting=False, run_random_forest=False, run_nn=False):
   
    print('will load data')

    # Carrega dados
    raw_data = load_lactic_acid_production_data()

    X, y = raw_data.generate_X_and_y(features_names=[
            'x_time_elapsed_hours',
            'x_lactobacillus_casei',
            'x_l_lactic_acid',
            'x_whey_lactose',
    ], 
    targets_names=[
            'y_lactobacillus_casei',
            'y_l_lactic_acid',
            'y_whey_lactose',
    ])

    print(f'initial X shape = {X.shape}')
    print(f'initial y shape = {y.shape}')
    print(f'X contains nan = {np.any(np.isnan(X))}')
    print(np.argwhere(np.isnan(X)))
    print(f'y contains nan = {np.any(np.isnan(y))}')
    print(np.argwhere(np.isnan(y)))
   
    # Removendo as labels solicitadas
    raw_data.dict = simple_label_remover(
        dict=raw_data.dict,
        labels_to_remove= {
        ## Remove todos os itens onde a coluna reference contém o valor manual_adjustments
        'data_origin':['manual'],
        'data_type': ['manual'],
        ## Remove todos os itens onde a coluna is_noise contém o valor 1 = True
        'is_noise': [1]
        }   
        )
    
    print('label removing finished')

    # Separa X e y que serão usados, ignora o resto
    X, y = raw_data.generate_X_and_y(
        features_names=[
            'x_time_elapsed_hours',
            'x_lactobacillus_casei',
            'x_l_lactic_acid',
            'x_whey_lactose',
    ], 
    targets_names=[
            'y_lactobacillus_casei',
            'y_l_lactic_acid',
            'y_whey_lactose',
    ]
    )

    # TODO será que eu realmente devia ter usado todos os valores da simulação?
    # daí ao invés de 800 pontos teria uns 5-6k (pulei de 4 em 4 ou 8 em 8 dependendo do caso!)
    # isso pode melhorar bastante os resultados...

    # Para ranges de dados MUITO grandes assim, o erro aumenta mto. Para ranges baixo fica
    # overfitted
    # X = X[0:999,:]
    # y = y[0:999,:]
    X = X[000:1200]
    y = y[000:1200]

    # TODO usa o label filter pra pegar só o X e y de um único ref/subref (altiok 2006 fig2) e
    # plot só ele usando "s" e "e"
    # prevê o básico e plota, dados 250-350
    s = 100; # range start
    e = 650; # range end
    
    # TODO erro nan????? Eu cropei pra resolver, mas o certo é descobrir onde tá
    # TODO printar time, X,P,S entrada. Tem algo errado. Não tá fazendo sentido esses
    # valores de entrada...
    
    print(f'X shape = {X.shape}')
    print(f'y shape = {y.shape}')
    
    if(run_decision_trees):
        print('\n------------------------\n')
        print('starting decision trees')
        # DECISION TREES
        model = classic_decision_tree_loop(X, y, test_sizes=[0.1])

        prediction = model.predict(X[s:e])
        plotvalues(X=X[s:e], y=y[s:e], prediction=prediction, show_X=True)
        plotvalues(X=X[s:e], y=y[s:e], prediction=prediction, show_P=True)
        plotvalues(X=X[s:e], y=y[s:e], prediction=prediction, show_S=True)

        # Ada boost é unidim, nem plote agora n adianta
        # model = adaboost_decision_tree_loop(X, y)


    if(run_neigs):
        print('\n------------------------\n')
        print('starting kneigs')
        # K NEIG
        basic_k_neig_loop(X, y)

    if(run_gradient_boosting):
        raise ValueError(':::  GRADIENT BOOSTING ::: NOT IMPLEMENTED  :::');
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
    
    if(run_random_forest):
        # FIXME a random tá demorando MUITO pra otimizar com o gridsearch
        print('\n------------------------\n')
        print('starting random forest')
        model = basic_random_forest_loop(X, y)

        # print(f'best acc = {acc}')
        prediction = model.predict(X[s:e])
        plotvalues(X=X[s:e], y=y[s:e], prediction=prediction, show_X=True)
        plotvalues(X=X[s:e], y=y[s:e], prediction=prediction, show_P=True)
        plotvalues(X=X[s:e], y=y[s:e], prediction=prediction, show_S=True)

    if(run_nn):
        # TODO roda só com uma fração dos dados pq são muitos, vai passar a vida
        # SCIKIT
        model = scikit_nn_loop(X, y, debug_print=False)
        prediction = model.predict(X[s:e])
        plotvalues(X=X[s:e], y=y[s:e], prediction=prediction, show_X=True)
        plotvalues(X=X[s:e], y=y[s:e], prediction=prediction, show_P=True)
        plotvalues(X=X[s:e], y=y[s:e], prediction=prediction, show_S=True)
        
        # TODO faz um modelo com os valores padrão mantidos e variando o tempo
        # Assim consigo ver se ele entende que tem um ponto de máximo e dali não passa

        model = pytorch_nn_loop(X, y, debug_print=False)
        
        
    

    # TODO acho que todos os tipos de regrressão podem ficar num objeto
    # e chamar dele. Algo como "Fitter"
    # TODO gráfico numero de testes e depth vs precisão
    

    # TODO veja como fazer hot encoding


    pass


if __name__ == '__main__':
    main(run_decision_trees=False, run_neigs=False, run_gradient_boosting=True, run_random_forest=False, run_nn=True);