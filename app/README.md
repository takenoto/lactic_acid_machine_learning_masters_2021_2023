
# Exemplo
 - https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_training_curves.html#sphx-glr-auto-examples-neural-networks-plot-mlp-training-curves-py
 É assim que tem que ficar o que vou fazer, com uma série de parâmetros passados por fora. Se não, vai virar bagunça.


## Explicações
 - Discrete vs continuous outputs [http://web.science.mq.edu.au/~mjohnson/papers/Johnson14-02ML-talk.pdf]
    * classification: supervised learning with discrete outputs
    * regression: supervised learning with continuous outputs
    * clustering: unsupervised learning with discrete outputs
    * dimensionality reduction: unsupervised learning with continuous outputs


# Falta estudar
- https://scikit-learn.org/stable/modules/preprocessing.html
- https://scikit-learn.org/stable/modules/neural_networks_supervised.html#complexity
- https://www.relataly.com/stock-price-prediction-multi-output-regression-using-neural-networks-in-python/5800/
- Livro na pasta mestrado/tcc
    - Feature importance
- https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html#sphx-glr-auto-examples-model-selection-plot-nested-cross-validation-iris-py
- https://scikit-learn.org/stable/modules/neighbors.html
- https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_training_curves.html#sphx-glr-auto-examples-neural-networks-plot-mlp-training-curves-py
- https://scikit-learn.org/stable/auto_examples/index.html
- https://www.frontiersin.org/files/Articles/490924/fbioe-08-00007-HTML/image_m/fbioe-08-00007-g001.jpg
- https://pubmed.ncbi.nlm.nih.gov/18634039/
- https://www.google.com/search?q=Lactic+acid+production+from+lactose+by+Lactobacillus+plantarum%3A+kinetic+model+and+effects+of+pH%2C+substrate%2C+and+oxygen.&oq=Lactic+acid+production+from+lactose+by+Lactobacillus+plantarum%3A+kinetic+model+and+effects+of+pH%2C+substrate%2C+and+oxygen.&aqs=chrome..69i57.607j0j1&sourceid=chrome&ie=UTF-8


## Todos
- Criar pasta só com excel com simulações POR ARTIGO. Aí depois copio esses dados e colo no banco padrão.
- Preciso MUITO adicionar coisas de outros artigos (dados experimentais), dados de modelagem e fazer a comparação, ver se são o suficiente pra predição, etc.
- Os dados manuais tb devo gerar uma lista, por exemplo, de forma que para qualquer valor de pH ou T, quando tudo for 0, tudo permanece "0" pra que ele aprenda que a massa tem que sair de algum canto.
- LEr livro oreilly -> trees + NN
- Ver como criar aquelas funções de classificação e orientação, pq aí posso manualmente aumentar o erro quando a conc < 0 pra simbolizar que nunca deve ser < 0.
- Modelos a serem criados:
    *Gridsearch for parameter tuning ???
    * Essa separação que to fazendo na marra em loops (% test e % train) for daria pra fazer usando um kfold validation??
        - randozimzação do k se for true gera valores que dificilmente vou reproduzir depois, melhor passar um int
    * Pipeline (ver 3 docs + artigo medium)
    * k nearest neighbour regression (ver vídeo e docs)
        --> Manifold learning
    * Random Forest (ver vídeo e docs)
    * Neural Networks
    * Gradient Boost
    * Tenta novamente Adaboost, mas como ele é 1D, faz um adaboost pra cada variável de saída (LLA, X e S)
    * Bayesiano simples, vale a pena?
    * SVM (acho que tb é 1D, veja os exemplos dos artigos. Aí não adianta querer usar ele aqui.)
- Criar mais dados, a partir de modelos matemáticos se for o caso, para preencher.
- fazer testes considerando on/off com dados *de ajuste manual*, provenientes de *equações*, *ruídos (is_noise)* e afins. Aí sabemos se vale à pena inserção desses dados, o quanto eles melhoram ou pioram as predições e se é necessário pegar muitos dados experimentais.
- Adicionar mais dados de outras bactérias (após conseguir estabilizar a L. casei com sucesso)

### Sobre o fit
- Quando o shuffle=True, depende MUITO da interação a precisão. Não significa que precise necessariamente desativar, mas como está (76 pontos de dados) não tem nem perigo de dar certo. Às vezes fica um ajuste muito bom mas depende de aleatoriedade.
- Se desativar o shuffle, os dados adicionados no final começam a ser menos relevantes em função dos dados do início serem usados pra fit.


### Sobre dados
- Os dados serem randomizados ao inserir pode causar problemas. Veja:
    * Se os dados são ao longo do tempo e na randomização os dados nos tempos finais e iniciais acabarem sendo excluídos, o sistema vai ter apenas os valores intermediários e não vai prever
    corretamente a tendência de estacionar em concentrações muito baixas e afins.
- O uso de modelos para "fechar" essas lacunas gerando uma série de dados pode ser bem útil, bem como misturar eles com dados reais.