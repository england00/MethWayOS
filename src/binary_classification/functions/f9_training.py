import time


def training(dataframe, models):
    """
        :param dataframe: training set loaded inside a dataframe
        :param models: dictionary of models to train
        :return model: dictionary of trained models
    """
    # Splitting FEATURES and LABELS
    X = dataframe.drop('y', axis=1)
    y = dataframe['y']

    # Training each model with Training Set
    for item in models:
        set_t0 = time.time()
        models[item].fit(X, y)
        print(f'{item.upper()}\' training took {time.time() - set_t0} sec')

    return models
