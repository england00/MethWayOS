import pandas as pd
from sklearn.preprocessing import StandardScaler


def features_preprocessing(dataframe, lower_threshold=0, upper_threshold=0, verbose=True, column_names=None):
    """
        :param dataframe: dataset loaded inside a dataframe
        :param lower_threshold: threshold for DEAD cases
        :param upper_threshold: threshold for ALIVE cases
        :param verbose: printing selector for additional data on STDOUT
        :param column_names: dataframe columns
        :return dataframe: preprocessed dataset loaded inside a dataframe
    """
    if verbose:
        # Checking possible values problems
        print('NAN VALUES:\n', dataframe.isnull().any())
        print('\nMISSING VALUES:\n', (dataframe == ' ').any())
        print('\nDUPLICATED VALUES:\n', dataframe.duplicated(keep='first'), '\n')

    # Managing imposed thresholds
    if lower_threshold != 0 and upper_threshold != 0:
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
    dataframe.loc[dataframe['y'] <= lower_threshold, 'y'] = 1  # changing lower values with '1'
    dataframe.loc[dataframe['y'] >= upper_threshold, 'y'] = 2  # changing higher values with '2'

    # Selecting only rows with labels '1' and '2'
    label_values = [1, 2]
    dataframe = dataframe[dataframe['y'].isin(label_values)]

    # Feature Scaling
    X = dataframe.drop('y', axis=1)
    y = dataframe['y']
    scaler = StandardScaler().fit(X.astype('float64'))
    X = pd.DataFrame(scaler.transform(X[column_names].astype('float64')),
                     columns=column_names,
                     index=X[column_names].index)

    return pd.concat([X, y], axis=1, sort=False)
