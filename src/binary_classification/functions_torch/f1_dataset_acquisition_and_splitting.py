from data.methods.csv_dataset_loader import csv_loader
from sklearn.model_selection import train_test_split


def dataset_acquisition_and_splitting(path, shuffle, rand_state, lower_threshold=None, upper_threshold=None):
    """
        :param path: dataset directory
        :param lower_threshold: threshold for DEAD cases
        :param upper_threshold: threshold for ALIVE cases
        :param shuffle: shuffle flag
        :param rand_state: chosen random seed
        :return training_dataframe: training set loaded inside a dataframe
        :return testing_dataframe: testing set loaded inside a dataframe
        :return dataframe_columns: columns of the obtained dataframe
    """
    # Acquiring data from CSV file
    dataframe, dataframe_columns = csv_loader(path)

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

    # Splitting dataset in TRAINING and TESTING
    training_dataframe, testing_dataframe = train_test_split(dataframe,
                                                             test_size=0.2,
                                                             random_state=rand_state,
                                                             stratify=dataframe['y'],
                                                             shuffle=shuffle)
    print('DIMENSIONS:')
    print(f"\t--> Training Set: {len(training_dataframe)}")
    print(f"\t--> Testing Set: {len(testing_dataframe)}")

    return training_dataframe, testing_dataframe, dataframe_columns
