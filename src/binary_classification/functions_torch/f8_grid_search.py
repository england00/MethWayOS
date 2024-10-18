import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from src.binary_classification.functions_torch.f9_mlp_models import *


def grid_search(device, x, y, shuffle, rand_state, hyperparameters, k_folds=5):
    """
        :param device: CPU or GPU
        :param x: training set without labels
        :param y: training set with only labels
        :param shuffle: shuffle flag
        :param rand_state: chosen random seed
        :param hyperparameters: dictionary of all the possible hyperparameters to test the model with
        :param k_folds: number of used folds for cross validation
        :return: best_parameters (dictionary of hyperparameters), best_validation_loss (float), best_accuracy (float)
        :return: k_fold: current k_fold setting
    """
    # Setting K-Fold Cross Validation
    k_fold = KFold(n_splits=k_folds, shuffle=shuffle, random_state=rand_state)

    # Best Params Initialization
    best_mean_accuracy = 0.0
    best_mean_validation_loss = float('inf')
    best_parameters = {}

    # Grid Search
    for hidden_sizes in hyperparameters["hidden_layers_configuration"]:
        for learning_rate in hyperparameters["learning_rate"]:
            for batch_size in hyperparameters["batch_size"]:
                for alpha in hyperparameters["alpha"]:
                    for dropout in hyperparameters["dropout"]:
                        fold_accuracies = []
                        fold_validation_losses = []

                        # Cross Validation
                        for training_index, validation_index in k_fold.split(x):
                            # Validation Set Folds and GPU moving
                            X_fold_training, X_fold_validation = (x[training_index],
                                                                  x[validation_index])
                            y_fold_training, y_fold_validation = (y[training_index],
                                                                  y[validation_index])
                            X_fold_training, X_fold_validation = (X_fold_training.to(device),
                                                                  X_fold_validation.to(device))
                            y_fold_training, y_fold_validation = (y_fold_training.to(device),
                                                                  y_fold_validation.to(device))

                            # MLP Model for this fold
                            training_set = TensorDataset(X_fold_training, y_fold_training)
                            training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=shuffle)
                            model = MLPHidden(input_size=x.shape[1],
                                              hidden_layer_config=hidden_sizes,
                                              output_size=2,
                                              dropout_rate=dropout).to(device)
                            criterion = nn.CrossEntropyLoss()
                            optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(alpha, 0.999))

                            # Early Stopping Parameters
                            early_stopping_patience = 2
                            best_validation_loss = float('inf')
                            counter = 0

                            # Training Model
                            for epoch in range(hyperparameters["max_epochs_number"]):
                                model.train()
                                for X_batch, y_batch in training_loader:
                                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                                    # Forward Pass
                                    outputs = model(X_batch)
                                    loss = criterion(outputs, y_batch)

                                    # Backward Pass
                                    optimizer.zero_grad()
                                    loss.backward()
                                    optimizer.step()

                                # Model Evaluation
                                model.eval()
                                with torch.no_grad():
                                    validation_outputs = model(X_fold_validation)
                                    validation_loss = criterion(validation_outputs, y_fold_validation).item()

                                # Early Stopping check
                                if validation_loss < best_validation_loss:
                                    best_validation_loss = validation_loss
                                    counter = 0
                                else:
                                    counter += 1
                                    if counter >= early_stopping_patience:
                                        break

                            # Final Model Evaluation on Validation Set
                            model.eval()
                            with torch.no_grad():
                                outputs = model(X_fold_validation)
                                _, predicted = torch.max(outputs, 1)
                                accuracy = (predicted == y_fold_validation).sum().item() / y_fold_validation.size(0)
                                fold_accuracies.append(accuracy)
                                fold_validation_losses.append(best_validation_loss)

                        # Mean Accuracy and Mean Validation Loss
                        mean_accuracy = sum(fold_accuracies) / len(fold_accuracies)
                        mean_validation_loss = sum(fold_validation_losses) / len(fold_validation_losses)

                        # Saving Best Configuration
                        # Update the best parameters if both Mean Accuracy is higher and Mean Validation loss is lower
                        if mean_accuracy > best_mean_accuracy and mean_validation_loss < best_mean_validation_loss:
                            best_parameters = {
                                'hidden_layers_configuration': hidden_sizes,
                                'learning_rate': learning_rate,
                                'batch_size': batch_size,
                                'alpha': alpha,
                                'dropout': dropout,
                                'max_epochs_number': hyperparameters["max_epochs_number"]
                            }
                            best_mean_accuracy = mean_accuracy
                            best_mean_validation_loss = mean_validation_loss

                        print(f'\t--> Hidden Size: {hidden_sizes}, '
                              f'Learning Rate: {learning_rate}, '
                              f'Batch Size: {batch_size}, '
                              f'Alpha: {alpha}, '
                              f'Dropout: {dropout}, '
                              f'Mean Accuracy: {mean_accuracy:.4f}, '
                              f'Mean Validation Loss: {mean_validation_loss:.4f}, ',
                              f'Current Best Mean Accuracy: {best_mean_accuracy:.4f}, ',
                              f'Current Best Mean Validation Loss: {best_mean_validation_loss:.4f}')

    print('\nBest value for each hyperparameter:')
    print(f'\t--> Hidden Size: {best_parameters["hidden_layers_configuration"]},\n'
          f'\t--> Learning Rate: {best_parameters["learning_rate"]},\n'
          f'\t--> Batch Size: {best_parameters["batch_size"]},\n'
          f'\t--> Alpha: {best_parameters["alpha"]},\n'
          f'\t--> Dropout: {best_parameters["dropout"]}')
    print(f'Best Mean Accuracy: {best_mean_accuracy:.4f},\n'
          f'Best Mean Validation Loss: {best_mean_validation_loss:.4f}')

    return best_parameters, k_fold
