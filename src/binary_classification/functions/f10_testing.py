import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def testing(dataframe, models):
    """
        :param dataframe: testing set loaded inside a dataframe
        :param models: dictionary of models to test
    """
    # Splitting FEATURES and LABELS
    X = dataframe.drop('y', axis=1)
    y = dataframe['y']

    # Testing each model with Testing Set and printing metrics
    y_pred = {}
    for item in models:
        set_t0 = time.time()
        y_pred[item] = models[item].predict(X)
        print(f'{item.upper()}\' testing took {time.time() - set_t0} sec:')
        print('\t--> Accuracy: ', accuracy_score(y, y_pred[item]))
        print('\t--> Precision: ', precision_score(y, y_pred[item], average='weighted'))
        print('\t--> Recall: ', recall_score(y, y_pred[item], average='weighted'))
        print('\t--> F1-Score: ', f1_score(y, y_pred[item], average='weighted'))
