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

## CONFIGURATION
JSON_PATHS_YAML = '../../config/files/json_paths.yaml'
DATASET_PATH_YAML = '../../config/files/dataset_paths.yaml'
METHYLATION = 'methylation'
METHYLATION_NAMES = 'methylation_names'
LOG_PATH = '../../logs/files/3.4 - METHYLATION & OS - Binary Classification (GPU).txt'
RANDOM_STATE = None  # if 'None' changes the seed to split training set and test set every time
LOWER_THRESHOLD = 1500  # 730 (2 years)
UPPER_THRESHOLD = 2500  # 2920 (8 years)
PCA_DIMENSION = 80
FEATURES_NUMBER = 20
VERBOSE = False
PLOT = False


## FUNCTIONS
def title(text):
    print(Fore.CYAN, '\n' + f' {text} '.center(80, '#'), Fore.RESET)


# CLASSES
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
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

    # Grid Search dei parametri
    hidden_sizes = [32, 64, 128]
    learning_rates = [0.01, 0.001, 0.0001]
    batch_sizes = [16, 32, 64]
    num_epochs = 20

    best_accuracy = 0.0
    best_params = {}

    for hidden_size in hidden_sizes:
        for lr in learning_rates:
            for batch_size in batch_sizes:
                # Creazione DataLoader
                train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)

                # Creazione e spostamento del modello su GPU
                model = MLP(input_size=training_set.drop('y', axis=1).shape[1], hidden_size=hidden_size,
                            output_size=2).to(device)

                # Definizione della funzione di perdita e dell'ottimizzatore
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)

                # Addestramento del modello
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

                # Valutazione del modello
                model.eval()
                with torch.no_grad():
                    outputs = model(X_testing_tensor)
                    _, predicted = torch.max(outputs, 1)
                    accuracy = (predicted == y_testing_tensor).sum().item() / y_testing_tensor.size(0)

                    # Salvataggio dei migliori parametri
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {
                            'hidden_size': hidden_size,
                            'learning_rate': lr,
                            'batch_size': batch_size
                        }

                print(
                    f'Hidden Size: {hidden_size}, Learning Rate: {lr}, Batch Size: {batch_size}, Accuracy: {accuracy:.4f}')

    # Stampa dei migliori parametri
    print("\nMigliori parametri trovati:")
    print(best_params)
    print(f'Migliore accuratezza: {best_accuracy:.4f}')
