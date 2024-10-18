from data.methods.csv_loader import csv_loader
from sklearn.model_selection import train_test_split


def dataset_acquisition_and_splitting(path, json_paths_yaml, names, shuffle, rand_state):
    """
        :param path: dataset directory
        :param json_paths_yaml: YAML file path to load
        :param names: features name list
        :param shuffle: shuffle flag
        :param rand_state: chosen random seed
        :return training_dataframe: training set loaded inside a dataframe
        :return testing_dataframe: testing set loaded inside a dataframe
        :return dataframe_columns: columns of the obtained dataframe
    """
    # Acquiring data from CSV file
    dataframe, dataframe_columns = csv_loader(path, json_paths_yaml, names)

    # Splitting dataset in TRAINING and TESTING
    training_dataframe, testing_dataframe = train_test_split(dataframe,
                                                             test_size=0.2,
                                                             random_state=rand_state,
                                                             shuffle=shuffle)
    print('DIMENSIONS:')
    print(f"\t--> Training Set: {len(training_dataframe)}")
    print(f"\t--> Testing Set: {len(testing_dataframe)}")

    return training_dataframe, testing_dataframe, dataframe_columns
