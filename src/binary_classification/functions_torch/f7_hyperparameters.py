def general_hyperparameters():
    """
        :return dictionary: dictionary of hyperparameters
    """
    '''
    return {
        "alpha": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        "batch_size": [4, 8, 16, 32, 64, 128],
        "dropout": [0.3, 0.5, 0.7],
        "hidden_layers_configuration": [[8], [16], [32], [8, 4], [16, 8], [32, 16]],
        "learning_rate": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        "weight_decay": [1e-1, 1e-2, 1e-3],
        "max_epochs_number": 100}
        '''

    return {
        "alpha": [1e-3, 1e-4, 1e-5],
        "batch_size": [8, 16, 32, 64],
        "dropout": [0.2, 0.4, 0.6],
        "hidden_layers_configuration": [[8], [16], [8, 4], [16, 8]],
        "learning_rate": [1e-3, 1e-4],
        "weight_decay": [1e-3, 1e-4],
        "max_epochs_number": 200}


def single_layer_hyperparameters():
    """
        :return dictionary: dictionary of hyperparameters
    """
    return {
        "alpha": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        "batch_size": [4, 8, 16, 32, 64, 128],
        "dropout": [0.3, 0.5, 0.7],
        "hidden_layers_configuration": [[8], [16], [32], [64]],
        "learning_rate": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        "weight_decay": [1e-1, 1e-2, 1e-3],
        "max_epochs_number": 100}


def double_layer_hyperparameters():
    """
        :return dictionary: dictionary of hyperparameters
    """
    return {
        "alpha": [0.01, 0.005, 0.001, 0.0001],
        "batch_size": [8, 16, 32, 64],
        "dropout": [0.3, 0.5, 0.7],
        "hidden_layers_configuration": [[8, 4], [16, 8], [32, 16], [64, 32], [128, 64]],
        "learning_rate": [0.01, 0.001, 0.0001],
        "weight_decay": [1e-1, 1e-2, 1e-3],
        "max_epochs_number": 100}


def three_layers_hyperparameters():
    """
        :return dictionary: dictionary of hyperparameters
    """
    return {
        "alpha": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        "batch_size": [4, 8, 16, 32, 64],
        "dropout": [0.0, 0.3, 0.5, 0.7],
        "hidden_layers_configuration": [[16, 8, 4], [32, 16, 8], [64, 32, 16]],
        "learning_rate": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        "weight_decay": [1e-1, 1e-2, 1e-3],
        "max_epochs_number": 100}


def four_layers_hyperparameters():
    """
        :return dictionary: dictionary of hyperparameters
    """
    return {
        "alpha": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        "batch_size": [4, 8, 16, 32, 64],
        "dropout": [0.0, 0.3, 0.5, 0.7],
        "hidden_layers_configuration": [[32, 16, 8, 4], [64, 32, 16, 8], [32, 64, 32, 16]],
        "learning_rate": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        "weight_decay": [1e-1, 1e-2, 1e-3],
        "max_epochs_number": 100}
