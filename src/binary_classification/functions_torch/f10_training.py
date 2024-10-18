import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from src.binary_classification.functions_torch.f9_mlp_models import *


def training(device, X, y, rand_state, hyperparameters, k_folds=5):
    """
        :param device: CPU or GPU
        :param X: training set without labels
        :param y: training set with only labels
        :param rand_state: chosen random seed
        :param hyperparameters: dictionary of all the possible hyperparameters to test the model with
        :param k_folds: number of used folds for cross validation
        :return model: trained model
    """
    # Setting K-Fold Cross Validation
    k_fold = KFold(n_splits=5, shuffle=True, random_state=rand_state)

    # Best Training Params Initialization
    best_model = None
    best_fold_loss = float('inf')

    # Cross Validation
    for fold, (training_index, validation_index) in enumerate(k_fold.split(X)):
        print(f'\nFold {fold + 1}/{k_folds}')

        # Validation Set Folds and GPU moving
        X_fold_training, X_fold_validation = (X[training_index],
                                              X[validation_index])
        y_fold_training, y_fold_validation = (y[training_index],
                                              y[validation_index])
        X_fold_training, X_fold_validation = (X_fold_training.to(device),
                                              X_fold_validation.to(device))
        y_fold_training, y_fold_validation = (y_fold_training.to(device),
                                              y_fold_validation.to(device))

        # Create DataLoader for training data
        training_set = TensorDataset(X_fold_training, y_fold_training)
        training_loader = DataLoader(training_set, batch_size=hyperparameters['batch_size'], shuffle=True)

        # MLP Model for this fold
        model = MLP2Hidden(input_size=X.shape[1],
                           hidden_layer_config=hyperparameters['hidden_layers_configuration'],
                           output_size=2,
                           dropout_rate=hyperparameters['dropout']).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),
                               lr=hyperparameters['learning_rate'],
                               betas=(hyperparameters['alpha'], 0.999))

        # Early Stopping Parameters
        patience = 5
        best_val_loss = float('inf')
        epochs_no_improve = 0

        # Learning Rate Scheduling
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        # Training Model for this fold
        set_t0 = time.time()
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

            # Model Evaluation on Validation Set
            model.eval()
            with torch.no_grad():
                outputs = model(X_fold_validation.to(device))
                val_loss = criterion(outputs, y_fold_validation.to(device)).item()
            print(f'\t--> Epoch [{epoch + 1}/{hyperparameters["max_epochs_number"]}], '
                  f'Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')

            # Learning Rate update
            scheduler.step()

            # Early Stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # torch.save(model.state_dict(), './model_weights/Methylation & OS Binary Classification.pth')
                best_model_fold = model  # Save the model of this fold
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping")
                    break

        # Keep track of the best model across all folds
        if best_val_loss < best_fold_loss:
            best_fold_loss = best_val_loss
            best_model = best_model_fold  # Update best model

        print(f'\t--> Training for Fold {fold + 1} took {time.time() - set_t0} sec')

    print(f'\nBest validation loss across all folds: {best_fold_loss:.4f}')
    return best_model
