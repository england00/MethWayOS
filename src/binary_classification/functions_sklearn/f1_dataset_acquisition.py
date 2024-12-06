from data.methods.csv_dataset_loader import csv_loader


def dataset_acquisition(path):
    """
        :param path: dataset directory
        :return dataframe: dataset loaded inside a dataframe
        :return dataframe_columns: columns of the obtained dataframe
    """
    dataframe, dataframe_columns = csv_loader(path)

    return dataframe, dataframe_columns
