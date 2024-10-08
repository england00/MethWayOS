import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.torch_tests.functions.f9_mlp_models import *


def training(device, X, y, X_validation, y_validation, hyperparameters):
    """
        :param device: CPU or GPU
        :param X: training set without labels
        :param y: training set with only labels
        :param X_validation: validation set without labels
        :param y_validation: validation set with only labels
        :param hyperparameters: dictionary of all the possible hyperparameters to test the model with
        :return model: trained model
    """
    # MLP Model
    training_set = TensorDataset(X, y)
    training_loader = DataLoader(training_set, batch_size=hyperparameters['batch_size'], shuffle=True)
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

    # Training Model
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

        # Model Evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_validation.to(device))
            val_loss = criterion(outputs, y_validation.to(device)).item()
        print(f'\t--> Epoch [{epoch + 1}/{hyperparameters["max_epochs_number"]}], '
              f'Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')

        # Learning Rate update
        scheduler.step()

        # Early Stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # torch.save(model.state_dict(), './model_weights/Methylation & OS Binary Classification.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping")
                break

    print(f'\t--> Training took {time.time() - set_t0} sec')
    return model
