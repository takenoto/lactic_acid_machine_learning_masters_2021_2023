# Foreign imports
import numpy as np

# Local imports
from files.files import load_lactic_acid_production_data;
from domain.data_dict_simple_label_remover import simple_label_remover


# Coisas para o loop local
from app.decision_tree.decision_tree import classic_decision_tree_loop, adaboost_decision_tree_loop

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

def main():
    
    # Carrega dados
    raw_data = load_lactic_acid_production_data()
   
    raw_data.dict = simple_label_remover(dict=raw_data.dict)
    
    # Separa X e y que serão usados, ignora o resto
    X, y = raw_data.generate_X_and_y(
        features_names=[
            'x_time_elapsed_hours',
            'x_lactobacillus_casei',
            'x_l_lactic_acid',
            'x_whey',
    ], 
    targets_names=[
            'y_lactobacillus_casei',
            'y_l_lactic_acid',
            'y_whey',
    ])
    
    print(f'X shape = {X.shape}')
    print(f'y shape = {y.shape}')
    
    classic_decision_tree_loop(X, y)

    # FIXME apaga
    # Parece que o ada é pra output único... 
    adaboost_decision_tree_loop(X, y)
    
    

    # TODO acho que todos os tipos de regrressão podem ficar num objeto
    # e chamar dele. Algo como "Fitter"
    # TODO gráfico numero de testes e depth vs precisão
    

    # TODO veja como fazer hot encoding


    pass


if __name__ == '__main__':
    main();