from data.methods.csv_dataset_loader import csv_loader


def dataset_acquisition(path, json_paths_yaml, names):
    """
        :param path: dataset directory
        :param json_paths_yaml: YAML file path to load
        :param names: features name list
        :return dataframe: dataset loaded inside a dataframe
        :return dataframe_columns: columns of the obtained dataframe
    """
    dataframe, dataframe_columns = csv_loader(path, json_paths_yaml, names)

    return dataframe, dataframe_columns
