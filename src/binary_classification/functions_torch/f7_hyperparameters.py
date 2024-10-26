def general_hyperparameters():
    """
        :return dictionary: dictionary of hyperparameters
    """
    '''
    return {
        "alpha": [1e-2, 1e-3, 1e-4, 1e-5],
        "batch_size": [4, 8, 16, 32],
        "dropout": [0.0, 0.25, 0.5],
        "hidden_layers_configuration": [[2], [4], [5], [8], [4, 2], [5, 3], [8, 4]],
        "learning_rate": [1e-2, 1e-3],
        "weight_decay": [1e-2, 1e-3, 1e-4, 1e-5],
        "max_epochs_number": 500}
        '''
    return {
        "alpha": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        "batch_size": [4, 8, 16, 32, 64],
        "dropout": [0.0, 0.25, 0.5],
        "hidden_layers_configuration": [[2], [4], [5], [8], [4, 2], [5, 3], [8, 4]],
        "learning_rate": [1e-1, 1e-2, 1e-3, 1e-4],
        "weight_decay": [1e-2, 1e-3, 1e-4, 1e-5],
        "max_epochs_number": 500}


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
