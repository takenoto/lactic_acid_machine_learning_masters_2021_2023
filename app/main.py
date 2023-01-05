from files.files import load_lactic_acid_production_data;

# TODO faz regressão decision_tree facílima
# só pra testar pegar esses dados em 3 pontos e plotar o gráfico,
# veja se dá pra trabalhar tranquilo nessa nomenclatura ou vai dar bode


# TODO os prefixos de entrada e saída
# os identificadores das substâncias
# isso tudo deve ficar salvo numa classe própria

# TODO # as features e targets dependem do modelo a ser analisado, daí não ficam nessa
# classe geral

def main():
    # TODO abre o arquivo direto daqui, veja se consegue
    data = load_lactic_acid_production_data()
    print(f'samples = {data.samples_count}')
    X_label, y_label, X, y = data.generate_X_and_y(features_names=['reference'], targets_names=['sub_reference'])
    print(f'headers = {data.get_headers()}')
    pass


if __name__ == '__main__':
    main();