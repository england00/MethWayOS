import torch


def sklearn_to_torch(training_dataframe, testing_dataframe):
    """
        :param training_dataframe: training set loaded inside a dataframe
        :param testing_dataframe: testing set loaded inside a dataframe
        :return device: GPU or CPU
        :return training_set: torch tensor training set
        :return testing_set: torch tensor testing set
    """
    # GPU configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Training Set
    X_training_tensor = torch.tensor(training_dataframe.drop('y', axis=1).values, dtype=torch.float32).to(device)
    y_training_tensor = torch.tensor(training_dataframe['y'].values, dtype=torch.long).to(device)

    # Testing Set
    X_testing_tensor = torch.tensor(testing_dataframe.drop('y', axis=1).values, dtype=torch.float32).to(device)
    y_testing_tensor = torch.tensor(testing_dataframe['y'].values, dtype=torch.long).to(device)

    return device, X_training_tensor, y_training_tensor, X_testing_tensor, y_testing_tensor
