# python -m files.files

import numpy as np
import pandas as pd

# Na verdade a maior parte dos dados está separada em condição inicial e dados finais
# daí chamo os iniciais de x_ alguma coisa
# e os finais de y_ alguma coisa. Só isso, fim

class LoadedData:

    def get_headers(self):
        """
        Returns all headers of this data loaded.
        
        They can be X (features) or y (targets) but at this object
        they are not treated as different classes.
        """
        return list(self.dict.keys())


    def generate_X_and_y(self, features_names, targets_names):
        """
        Generates X and y (features and targets) input
        for machine learning algorithm.

        Returns the labels used, in order, for both x and y.

        X labels are the labels that will be loaded as the input
        y labels are the labels that will be loaded as the output
        everything else will be ignored
        """

        # Iterate over all Xs to get a list of list of X values
        X = [];
        for __X_label in features_names:
            X.append(self.dict[__X_label])
        # Iterate over all ys to get a list of list of y values
        y = [];
        for __y_label in targets_names:
            y.append(self.dict[__y_label])

        return features_names, targets_names, X, y;


    def __init__(self, dict):
        """
        dict is a dictionary that groups values per feature like this:
        {
            'feature_1': [value11, value12],
            'feature_2': [value22, value23]
        }
        """
        """
        Features are the columns names of the dataset.
        Features should not be edited directly, because the samples_count won't
        be calculated again. In this case, it is better to just create a new object.
        """
        self.dict=dict;
        """
        The samples themselves
        """
        self.samples_count=len(list(dict.values())[0]);
        """
        Number of samples (rows) in this dataset, excluding titles
        """


def load_lactic_acid_production_data():
    input_file = 'files/la_production_data.csv'
    df = pd.read_csv(input_file, header = 0, delimiter = ",")
    # Permite pegar cada propriedade, por coluna
    dict = df.to_dict('list')
    return LoadedData(dict=dict);

if __name__ == '__main__':
    load_lactic_acid_production_data();
