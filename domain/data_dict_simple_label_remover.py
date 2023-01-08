def simple_label_remover(dict, labels_to_remove:dict):
    """
    Remove todas as rows que contenham um valor definido na lista de valores da column (key)
    ex:
    `
        labels_to_remove = {
        
        # Remove todos os itens onde a coluna reference contém o valor manual_adjustments
        'reference':['manual_adjustments'],
       
        # Remove todos os itens onde a coluna is_noise contém o valor 1 = True
        'is_noise': [1]
        
        }
}
    `

    O dict original É MODIFICADO e RETORNADO (as 2 coisas).
    """
    for key in labels_to_remove:
        # 1º descobre os indexes dos marcados como manual e tira eles
        # r² tava dando -5, vamo ver agora
        column = dict[key]
        indexes_to_remove = []
        # i --> cada linha da coluna
        for i in range(len(column)):
            # Remove stamp é o valor ou string marcada como que aquela linha inteira deve
            # ser deletada
            for remove_stamp in labels_to_remove[key]:
                # O valor dessa linha da coluna
                row_value = column[i]
                if(row_value == remove_stamp):
                    if i not in indexes_to_remove:
                        indexes_to_remove.append(i)
        
        for key in dict:
            for index_to_remove in indexes_to_remove:
                del dict[key][index_to_remove]

    
    return dict;