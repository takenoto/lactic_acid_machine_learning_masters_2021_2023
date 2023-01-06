from domain.data_dealing.xy_data import XyData
from sklearn.model_selection import train_test_split

class DataSplitTestTrain:
    """
    Stores the test and train results for a Spli test
    """
    def __init__(self, train:XyData, test:XyData):
        self.train=train
        self.test=test

def split_data_for_test(X, y, test_size=0.2)->DataSplitTestTrain:
    """
    Returns a {DataSplitTestTrain}
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    train_data = XyData(X_train, y_train)
    test_data = XyData(X_test, y_test)

    return DataSplitTestTrain(train=train_data, test=test_data)

