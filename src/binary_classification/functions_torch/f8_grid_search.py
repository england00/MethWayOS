import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from src.binary_classification.functions_torch.f9_mlp_models import *
from src.binary_classification.functions_torch.f10_training_kfold_voting import training
from src.binary_classification.functions_torch.f11_testing_kfold_voting import testing


def grid_search(device, x, y, shuffle, rand_state, hyperparameters, k_folds, x_test, y_test):
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
                        for weight_decay in hyperparameters["weight_decay"]:
                            fold_accuracies = []
                            fold_validation_losses = []

                            # K-Fold Cross Validation with this Hyperparameters Configuration
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
                                training_loader = DataLoader(training_set,
                                                             batch_size=batch_size,
                                                             shuffle=shuffle)
                                model = MLPHidden(input_size=x.shape[1],
                                                  hidden_layer_config=hidden_sizes,
                                                  output_size=2,
                                                  dropout_rate=dropout).to(device)
                                criterion = nn.CrossEntropyLoss()
                                optimizer = optim.AdamW(model.parameters(),
                                                        lr=learning_rate,
                                                        betas=(alpha, 0.999),
                                                        weight_decay=weight_decay)

                                # Early Stopping Parameters
                                early_stopping_patience = 2
                                best_validation_loss = float('inf')
                                patience_counter = 0

                                # Training Model
                                for epoch in range(hyperparameters["max_epochs_number"]):
                                    model.train()
                                    for X_batch, y_batch in training_loader:
                                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                                        # Forward Pass
                                        final_validation_outputs = model(X_batch)
                                        loss = criterion(final_validation_outputs, y_batch)

                                        # Backward Pass
                                        optimizer.zero_grad()
                                        loss.backward()
                                        optimizer.step()

                                    # Model Evaluation only for this epoch
                                    model.eval()
                                    with torch.no_grad():
                                        validation_outputs = model(X_fold_validation)
                                        validation_loss = criterion(validation_outputs, y_fold_validation).item()

                                    # Early Stopping check
                                    if validation_loss < best_validation_loss:
                                        best_validation_loss = validation_loss
                                        patience_counter = 0
                                    else:
                                        patience_counter += 1
                                        if patience_counter >= early_stopping_patience:
                                            break

                                # Final Model Evaluation with this Hyperparameters Configuration on Validation Set
                                model.eval()
                                with torch.no_grad():
                                    final_validation_outputs = model(X_fold_validation)
                                    final_validation_loss = criterion(final_validation_outputs, y_fold_validation).item()
                                    _, predicted = torch.max(final_validation_outputs, 1)
                                    accuracy = (predicted == y_fold_validation).sum().item() / y_fold_validation.size(0)
                                    fold_accuracies.append(accuracy)
                                    fold_validation_losses.append(final_validation_loss)

                            # Mean Accuracy and Mean Validation Loss
                            mean_accuracy = sum(fold_accuracies) / len(fold_accuracies)
                            mean_validation_loss = sum(fold_validation_losses) / len(fold_validation_losses)

                            # Saving Best Hyperparameters Configuration
                            # Update the best parameters if both Mean Accuracy is higher and Mean Validation loss is lower
                            if (mean_validation_loss < best_mean_validation_loss or
                                    (mean_validation_loss < (best_mean_validation_loss + 0.1)
                                     and mean_accuracy > best_mean_accuracy)):
                                best_parameters = {
                                    'hidden_layers_configuration': hidden_sizes,
                                    'learning_rate': learning_rate,
                                    'batch_size': batch_size,
                                    'alpha': alpha,
                                    'dropout': dropout,
                                    'weight_decay': weight_decay,
                                    'max_epochs_number': hyperparameters["max_epochs_number"]
                                }
                                best_mean_accuracy = mean_accuracy
                                best_mean_validation_loss = mean_validation_loss

                                ###########################################################################
                                print('\n\nCURRENT BEST MODEL')
                                print(f'\t--> Hidden Size: {hidden_sizes}, '
                                      f'Learning Rate: {learning_rate}, '
                                      f'Batch Size: {batch_size}, '
                                      f'Alpha: {alpha}, '
                                      f'Dropout: {dropout}, '
                                      f'Weigh Decay: {weight_decay}, '
                                      f'Accuracy: {mean_accuracy:.4f}, '
                                      f'Validation Loss: {mean_validation_loss:.4f}, ',
                                      f'Current Best Accuracy: {best_mean_accuracy:.4f}, ',
                                      f'Current Best Validation Loss: {best_mean_validation_loss:.4f}')

                                # Training
                                print('\nTRAINING:')
                                current_model = training(
                                    device=device,
                                    x=x,
                                    y=y,
                                    shuffle=shuffle,
                                    hyperparameters=best_parameters,
                                    k_fold_setting=k_fold)

                                # Testing
                                print('\nTESTING:')
                                testing(
                                    models=current_model,
                                    x_testing=x_test,
                                    y_testing=y_test)

                                print('HYPERPARAMETERS:')
                                print(f'\t--> Hidden Size: {hidden_sizes}, '
                                      f'Learning Rate: {learning_rate}, '
                                      f'Batch Size: {batch_size}, '
                                      f'Alpha: {alpha}, '
                                      f'Dropout: {dropout}, '
                                      f'Weigh Decay: {weight_decay}, '
                                      f'Accuracy: {mean_accuracy:.4f}, '
                                      f'Validation Loss: {mean_validation_loss:.4f}, ',
                                      f'Current Best Accuracy: {best_mean_accuracy:.4f}, ',
                                      f'Current Best Validation Loss: {best_mean_validation_loss:.4f}')

                                ###########################################################################

                            '''
                            print(f'\t--> Hidden Size: {hidden_sizes}, '
                                  f'Learning Rate: {learning_rate}, '
                                  f'Batch Size: {batch_size}, '
                                  f'Alpha: {alpha}, '
                                  f'Dropout: {dropout}, '
                                  f'Weigh Decay: {weight_decay}, '
                                  f'Accuracy: {mean_accuracy:.4f}, '
                                  f'Validation Loss: {mean_validation_loss:.4f}, ',
                                  f'Current Best Accuracy: {best_mean_accuracy:.4f}, ',
                                  f'Current Best Validation Loss: {best_mean_validation_loss:.4f}')
                            '''

    print('\nBest value for each hyperparameter:')
    print(f'\t--> Hidden Size: {best_parameters["hidden_layers_configuration"]},\n'
          f'\t--> Learning Rate: {best_parameters["learning_rate"]},\n'
          f'\t--> Batch Size: {best_parameters["batch_size"]},\n'
          f'\t--> Alpha: {best_parameters["alpha"]},\n'
          f'\t--> Dropout: {best_parameters["dropout"]},\n'
          f'\t--> Weigh Decay: {best_parameters["weight_decay"]}')
    print(f'Best Accuracy: {best_mean_accuracy:.4f},\n'
          f'Best Validation Loss: {best_mean_validation_loss:.4f}')

    return best_parameters, k_fold
