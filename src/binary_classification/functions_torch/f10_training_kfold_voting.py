import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.binary_classification.functions_torch.f9_mlp_models import *


def training(device, x, y, shuffle, hyperparameters, sk_fold_setting):
    """
        :param device: CPU or GPU
        :param x: training set without labels
        :param y: training set with only labels
        :param shuffle: shuffle flag
        :param hyperparameters: dictionary of all the possible hyperparameters to test the model with
        :param sk_fold_setting: cross validation setting used during Grid Search
        :return ensemble_model: ensemble of trained models
    """
    # Best Training Params Initialization
    ensemble_model = []

    # Cross Validation
    for fold, (training_index, validation_index) in enumerate(sk_fold_setting.split(x.cpu().numpy(), y.cpu().numpy())):
        print(f'\nFold {fold + 1}/{sk_fold_setting.n_splits}')

        # Validation Set Folds and GPU moving
        X_fold_training, X_fold_validation = (x[training_index], x[validation_index])
        y_fold_training, y_fold_validation = (y[training_index], y[validation_index])
        X_fold_training, X_fold_validation = (X_fold_training.to(device), X_fold_validation.to(device))
        y_fold_training, y_fold_validation = (y_fold_training.to(device), y_fold_validation.to(device))

        # MLP Model for this fold
        training_set = TensorDataset(X_fold_training, y_fold_training)
        training_loader = DataLoader(training_set, batch_size=hyperparameters['batch_size'], shuffle=shuffle)
        model = MLPHidden(input_size=x.shape[1],
                          hidden_layer_config=hyperparameters['hidden_layers_configuration'],
                          output_size=2,
                          dropout_rate=hyperparameters['dropout']).to(device)
        class_weights = torch.tensor([len(y) / (2 * torch.sum(y == 0)),
                                      len(y) / (2 * torch.sum(y == 1))], device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(),
                                lr=hyperparameters['learning_rate'],
                                betas=(hyperparameters['alpha'], 0.999),
                                weight_decay=hyperparameters['weight_decay'])

        # Early Stopping Parameters
        early_stopping_patience = 3
        loss = None
        best_validation_loss = float('inf')
        patience_counter = 0
        fold_epoch_counter = 0

        # Learning Rate Scheduling
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

        # Training Model for this fold
        set_t0 = time.time()
        for epoch in range(hyperparameters["max_epochs_number"]):
            model.train()
            fold_epoch_counter += 1
            for X_batch, y_batch in training_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # Forward Pass
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                # Backward Pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Model Evaluation only for this epoch
            model.eval()
            with torch.no_grad():
                outputs = model(X_fold_validation)
                validation_loss = criterion(outputs, y_fold_validation).item()
            print(f'\t--> Epoch [{epoch + 1}/{hyperparameters["max_epochs_number"]}], '
                  f'Loss: {loss.item():.4f}, Validation Loss: {validation_loss:.4f}')

            # Learning Rate update
            scheduler.step(validation_loss)

            # Early Stopping check
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early stopping")
                    break

        # Save the best model for this fold
        ensemble_model.append(model)

        # Checking epochs used with this fold
        print(f'\t--> Training for Fold {fold + 1} took {time.time() - set_t0} sec, using {fold_epoch_counter} epochs')

    return ensemble_model
