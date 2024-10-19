import numpy as np
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.binary_classification.functions_torch.f9_mlp_models import *


def training(device, x, y, shuffle, hyperparameters, n_models=5, validation_split=0.2):
    """
        :param device: CPU or GPU
        :param x: training set without labels
        :param y: training set with only labels
        :param shuffle: shuffle flag
        :param hyperparameters: dictionary of all the possible hyperparameters to test the model with
        :param n_models: number of models to train in the bagging ensemble
        :param validation_split: proportion of data to use for validation
        :return ensemble_model: ensemble of trained models
    """
    # Best Training Params Initialization
    ensemble_model = []

    # Split the dataset into training and validation sets
    dataset_size = len(x)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    val_size = int(dataset_size * validation_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    x_val = x[val_indices].to(device)
    y_val = y[val_indices].to(device)

    for model_idx in range(n_models):
        print(f'\nTraining model {model_idx + 1}/{n_models}')

        # Randomly sample the training set with replacement
        sampled_indices = np.random.choice(train_indices, size=len(train_indices), replace=True)
        x_sampled = x[sampled_indices].to(device)
        y_sampled = y[sampled_indices].to(device)

        # MLP Model for this iteration
        training_set = TensorDataset(x_sampled, y_sampled)
        training_loader = DataLoader(training_set, batch_size=hyperparameters['batch_size'], shuffle=shuffle)
        model = MLPHidden(input_size=x.shape[1],
                          hidden_layer_config=hyperparameters['hidden_layers_configuration'],
                          output_size=2,
                          dropout_rate=hyperparameters['dropout']).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(),
                                lr=hyperparameters['learning_rate'],
                                betas=(hyperparameters['alpha'], 0.999),
                                weight_decay=hyperparameters['weight_decay'])

        # Early Stopping Parameters
        early_stopping_patience = 3
        best_validation_loss = float('inf')
        patience_counter = 0

        # Training Model
        set_t0 = time.time()
        loss = None
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

            # Model Evaluation on validation set
            model.eval()
            with torch.no_grad():
                outputs = model(x_val)
                validation_loss = criterion(outputs, y_val).item()
            print(f'\t--> Epoch [{epoch + 1}/{hyperparameters["max_epochs_number"]}], '
                  f'Loss: {loss.item():.4f}, Validation Loss: {validation_loss:.4f}')

            # Early Stopping check
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early stopping")
                    break

        # Save the best model for this iteration
        ensemble_model.append(model)

        # Checking time used with this iteration
        print(f'\t--> Training for model {model_idx + 1} took {time.time() - set_t0} sec')

    return ensemble_model
