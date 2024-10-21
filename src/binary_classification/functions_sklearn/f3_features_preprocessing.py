import pandas as pd
from sklearn.preprocessing import StandardScaler


def features_preprocessing(dataframe, column_names, lower_threshold=None, upper_threshold=None, verbose=True):
    """
        :param dataframe: dataset loaded inside a dataframe
        :param column_names:  columns
        :param lower_threshold: threshold for DEAD cases
        :param upper_threshold: threshold for ALIVE cases
        :param verbose: printing selector for additional data on STDOUT
        :return dataframe: preprocessed dataset loaded inside a dataframe
    """
    if verbose:
        # Checking possible values problems
        print('NAN VALUES:\n', dataframe.isnull().any())
        print('\nMISSING VALUES:\n', (dataframe == ' ').any())
        print('\nDUPLICATED VALUES:\n', dataframe.duplicated(keep='first'), '\n')

    # Managing imposed thresholds
    if lower_threshold is not None and upper_threshold is not None:
        i = j = 0
        for item in dataframe['y']:
            if item <= lower_threshold:
                i += 1
            if item >= upper_threshold:
                j += 1
        print('ADMITTED SAMPLES VALUES:')
        print(f'\t--> {i} samples with a label lower than {lower_threshold}')
        print(f'\t--> {j} samples with a label bigger than {upper_threshold}')

        # Operations to do with BOTH TRAINING SET & TEST SET
        dataframe.loc[dataframe['y'] <= lower_threshold, 'y'] = 0  # changing lower values with '0'
        dataframe.loc[dataframe['y'] >= upper_threshold, 'y'] = 1  # changing higher values with '1'

        # Selecting only rows with labels '0' and '1'
        label_values = [0, 1]
        dataframe = dataframe[dataframe['y'].isin(label_values)]

    # Feature Scaling
    X = dataframe.drop('y', axis=1)
    y = dataframe['y']
    scaler = StandardScaler().fit(X.astype('float64'))
    X = pd.DataFrame(scaler.transform(X[column_names].astype('float64')),
                     columns=column_names,
                     index=X[column_names].index)

    return pd.concat([X, y], axis=1, sort=False)
