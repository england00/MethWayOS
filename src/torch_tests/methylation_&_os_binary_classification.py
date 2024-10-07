from colorama import Fore
from config.methods.configuration_loader import yaml_loader
from src.binary_classification.functions.f1_dataset_acquisition import dataset_acquisition
from src.binary_classification.functions.f2_exploratory_data_analysis import exploratory_data_analysis
from src.binary_classification.functions.f3_features_preprocessing import features_preprocessing
from src.binary_classification.functions.f4_feature_selection import feature_selection
from src.binary_classification.functions.f5_dataset_splitting import dataset_splitting
from src.torch_tests.functions.f6_sklearn_to_torch import sklearn_to_torch
from logs.methods.log_storer import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import KFold

## CONFIGURATION
JSON_PATHS_YAML = '../../config/files/json_paths.yaml'
DATASET_PATH_YAML = '../../config/files/dataset_paths.yaml'
METHYLATION = 'methylation'
METHYLATION_NAMES = 'methylation_names'
LOG_PATH = '../../logs/files/3.4 - METHYLATION & OS - Binary Classification (GPU).txt'
RANDOM_STATE = 42  # if 'None' changes the seed to split training set and test set every time
LOWER_THRESHOLD = 1000  # 730 (2 years)
UPPER_THRESHOLD = 3000  # 2920 (8 years)
PCA_DIMENSION = 80
FEATURES_NUMBER = 11
VERBOSE = False
PLOT = False


## FUNCTIONS
def title(text):
    print(Fore.CYAN, '\n' + f' {text} '.center(80, '#'), Fore.RESET)


# CLASSES
'''
class MLP1Hidden(nn.Module):
    def __init__(self, input_size, hidden_layer_config, output_size, dropout_rate):
        super(MLP1Hidden, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer_config[0])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_layer_config[0], output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x
'''
'''
class MLP2Hidden(nn.Module):
    def __init__(self, input_size, hidden_layer_config, output_size, dropout_rate):
        super(MLP2Hidden, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer_config[0])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_layer_config[0], hidden_layer_config[1])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_layer_config[1], output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
'''

'''
class MLP3Hidden(nn.Module):
    def __init__(self, input_size, hidden_layer_config, output_size, dropout_rate):
        super(MLP3Hidden, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer_config[0])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_layer_config[0], hidden_layer_config[1])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_layer_config[1], hidden_layer_config[2])
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(hidden_layer_config[2], output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x
'''

'''
class MLP4Hidden(nn.Module):
    def __init__(self, input_size, hidden_layer_config, output_size, dropout_rate):
        super(MLP4Hidden, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer_config[0])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_layer_config[0], hidden_layer_config[1])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_layer_config[1], hidden_layer_config[2])
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(hidden_layer_config[2], hidden_layer_config[3])
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc5 = nn.Linear(hidden_layer_config[3], output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        x = self.fc5(x)
        return x
'''


class MLP(nn.Module):
    def __init__(self, input_size, hidden_layer_config, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer_config[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_config[0], hidden_layer_config[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_layer_config[1], output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


## MAIN
if __name__ == "__main__":
    # Open LOG file
    logfile = open(LOG_PATH, 'w')
    sys.stdout = DualOutput(sys.stdout, logfile)

    # Data Acquisition
    title('DATA ACQUISITION')
    dataset_paths = yaml_loader(DATASET_PATH_YAML)
    dataset, dataset_columns = dataset_acquisition(
        path=dataset_paths[METHYLATION],
        json_paths_yaml=JSON_PATHS_YAML,
        names=METHYLATION_NAMES)

    # Exploratory Data Analysis
    title('EXPLORATORY DATA ANALYSIS with RAW DATA')
    exploratory_data_analysis(dataframe=dataset, verbose=VERBOSE, plot=PLOT)

    # Feature Preprocessing
    title('FEATURES PREPROCESSING')
    dataset_columns.remove('y')
    dataset = features_preprocessing(
        dataframe=dataset,
        lower_threshold=LOWER_THRESHOLD,
        upper_threshold=UPPER_THRESHOLD,
        verbose=VERBOSE,
        column_names=dataset_columns)

    # Feature Selection
    title('FEATURE SELECTION')
    dataset = feature_selection(
        dataframe=dataset,
        rand_state=RANDOM_STATE,
        pca_dimension=PCA_DIMENSION,
        feature_number=FEATURES_NUMBER)

    # Dataset Splitting
    title('DATASET SPLITTING')
    training_set, testing_set = dataset_splitting(dataframe=dataset, rand_state=RANDOM_STATE)

    # SKLearn to Torch
    title('SKLEARN TO TORCH')
    device, X_training_tensor, y_training_tensor, X_testing_tensor, y_testing_tensor = sklearn_to_torch(
        training_dataframe=training_set,
        testing_dataframe=testing_set)

    training_set = TensorDataset(X_training_tensor, y_training_tensor)

    #################################

    # GRID SEARCH dei parametri
    hidden_layer_configurations = [
        [10, 5], [5, 4]
    ]
    learning_rates = [0.01, 0.001, 0.0001]
    batch_sizes = [4, 8, 16, 32, 64]
    num_epochs = 30
    k_fold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    alphas = [0.01, 0.005, 0.001, 0.0001]

    best_accuracy = 0.0
    best_params = {}

    # Grid Search
    for hidden_sizes in hidden_layer_configurations:
        for lr in learning_rates:
            for batch_size in batch_sizes:
                for alpha in alphas:
                    fold_accuracies = []

                    for train_index, val_index in k_fold.split(X_training_tensor):
                        # Creazione dei set di addestramento e validazione
                        X_fold_train, X_fold_val = X_training_tensor[train_index], X_training_tensor[val_index]
                        y_fold_train, y_fold_val = y_training_tensor[train_index], y_training_tensor[val_index]

                        # Spostare i tensori su GPU
                        X_fold_train, X_fold_val = X_fold_train.to(device), X_fold_val.to(device)
                        y_fold_train, y_fold_val = y_fold_train.to(device), y_fold_val.to(device)

                        # Creazione DataLoader per batch
                        train_dataset = TensorDataset(X_fold_train, y_fold_train)
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                        # Creazione e spostamento del modello su GPU
                        model = MLP(input_size=X_training_tensor.shape[1],
                                    hidden_layer_config=hidden_sizes,
                                    output_size=2).to(device)

                        # Definizione della funzione di perdita e dell'ottimizzatore con lr adattivo
                        criterion = nn.CrossEntropyLoss()
                        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(alpha, 0.999))

                        # Parametri per Early Stopping
                        patience = 2  # Numero di epoche senza miglioramento
                        best_loss = float('inf')
                        counter = 0  # Contatore per le epoche senza miglioramento

                        # Addestramento del modello
                        for epoch in range(num_epochs):
                            model.train()
                            for X_batch, y_batch in train_loader:
                                # Spostare il batch su GPU
                                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                                # Forward pass
                                outputs = model(X_batch)
                                loss = criterion(outputs, y_batch)

                                # Backward pass e ottimizzazione
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()

                            # Valutazione del modello su validation set
                            model.eval()
                            validation_loss = 0.0
                            with torch.no_grad():
                                val_outputs = model(X_fold_val)
                                val_loss = criterion(val_outputs, y_fold_val)
                                validation_loss = val_loss.item()

                            # Controllo dell'early stopping
                            if validation_loss < best_loss:
                                best_loss = validation_loss
                                counter = 0  # Reset del contatore
                                # Salva i migliori pesi del modello
                                torch.save(model.state_dict(), 'best_model.pth')
                            else:
                                counter += 1  # Incrementa il contatore

                            # Verifica se il contatore supera la pazienza
                            if counter >= patience:
                                break

                        # Valutazione finale del modello su validation set
                        model.eval()
                        with torch.no_grad():
                            outputs = model(X_fold_val)
                            _, predicted = torch.max(outputs, 1)
                            accuracy = (predicted == y_fold_val).sum().item() / y_fold_val.size(0)
                            fold_accuracies.append(accuracy)

                    # Calcolo dell'accuratezza media dei fold
                    mean_accuracy = sum(fold_accuracies) / len(fold_accuracies)

                    # Salvataggio dei migliori parametri
                    if mean_accuracy > best_accuracy:
                        best_accuracy = mean_accuracy
                        best_params = {
                            'hidden_sizes': hidden_sizes,
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'alpha': alpha
                        }

                    print(
                        f'Hidden Size: {hidden_sizes}, Learning Rate: {lr}, Batch Size: {batch_size}, Alpha: {alpha}, Mean Accuracy: {mean_accuracy:.4f}')

    # Stampa dei migliori parametri
    print("\nMigliori parametri trovati:")
    print(best_params)
    print(f'Migliore accuratezza media: {best_accuracy:.4f}')

    ############ TRAINING
    print("\nAddestramento del modello con i migliori parametri trovati...")

    # Creazione DataLoader con i migliori parametri
    train_loader = DataLoader(training_set, batch_size=best_params['batch_size'], shuffle=True)

    # Creazione e spostamento del modello su GPU con i migliori parametri
    model = MLP(input_size=X_training_tensor.shape[1],
                hidden_layer_config=best_params['hidden_sizes'],
                output_size=2).to(device)

    # Definizione della funzione di perdita e dell'ottimizzatore con alpha
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'], betas=(best_params['alpha'], 0.999))

    # Aggiunta della logica di Early Stopping
    num_epochs = 100  # Un numero maggiore per avere spazio per l'early stopping
    patience = 5  # Numero di epoche senza miglioramenti da aspettare
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Aggiunta di uno schedulatore di learning rate (opzionale)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Addestramento del modello con i migliori parametri
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass e ottimizzazione
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Valutazione sul set di test per la perdita di validazione
        model.eval()
        with torch.no_grad():
            outputs = model(X_testing_tensor)
            val_loss = criterion(outputs, y_testing_tensor).item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss:.10f}')

        # Aggiornamento del learning rate
        scheduler.step()  # Aggiorna il learning rate

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping")
                break

    ############# TESTING

    # Valutazione del modello finale
    model.eval()
    with torch.no_grad():
        outputs = model(X_testing_tensor)
        _, predicted = torch.max(outputs, 1)
        final_accuracy = (predicted == y_testing_tensor).sum().item() / y_testing_tensor.size(0)
        print(f'\nAccuratezza finale del modello con i migliori parametri: {final_accuracy:.4f}')
