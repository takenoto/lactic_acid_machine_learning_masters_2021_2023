## Todos
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