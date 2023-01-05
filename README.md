
Application for ML applied to the production of lactic acid by different bacteria.

### Structure
- domain 
    * Principais classes, loops e coisas generalizadas que podem ser usada em qualquer ponto
- files
    * Arquivos como tabelas, bancos de dados e arquivos de imagem
- app
    * Funcionalidades específicas da aplicação, tais como:
        - Criação, ajuste, save e loading do modelo salvo (através de uma função comum) para que carregue os modelos de fora facilmente.
        - Ajustar modelo decision_tree (recebe lista de colunas a serem consideradas), bem como quais são input e quais são output (ou seja, recebe lista inputs e lista outputs indexes)
        - Ajustar modelo SVM (recebe lista de colunas a serem consideradas)
        - Rodar todos os modelos disponíveis e escolher o melhor
        - Rodar modelos e exibir gráficos

### Files convention
- Internal naming
    * Input or initial variables are used as "x", such as x_glucose for representation of glucose concentration.
    * Output or variables after time "dt" are prefixed with "_y", such as y_glucose for the concentration of glucose after a certain time in the given conditions.

# TODO
- Temperature and associated stuff (rpms, reactor volume) should be both at input and output IN THE DATA. If they won't be used in models is another question. For instance, having them as this will provide material to deal with cases that are not isothermal.