from colorama import Fore
from config.methods.configuration_loader import yaml_loader
from src.binary_classification.functions_torch.f1_dataset_acquisition_and_splitting import dataset_acquisition_and_splitting
from src.binary_classification.functions_sklearn.f2_exploratory_data_analysis import exploratory_data_analysis
from src.binary_classification.functions_sklearn.f3_features_preprocessing import features_preprocessing
from src.binary_classification.functions_torch.f4_features_selection_training_set import features_selection_for_training
from src.binary_classification.functions_torch.f5_features_selection_testing_set import features_selection_for_testing
from src.binary_classification.functions_torch.f6_sklearn_to_torch import sklearn_to_torch
from src.binary_classification.functions_torch.f7_hyperparameters import general_hyperparameters
from src.binary_classification.functions_torch.f8_grid_search import grid_search
from src.binary_classification.functions_torch.f10_training_kfold_voting import training
from src.binary_classification.functions_torch.f11_testing_kfold_voting import testing
from logs.methods.log_storer import *


## CONFIGURATION
JSON_PATHS_YAML = '../../config/files/json_paths.yaml'
DATASET_PATH_YAML = '../../config/files/dataset_paths.yaml'
METHYLATION = 'methylation'
METHYLATION_NAMES = 'methylation_names'
LOG_PATH = '../../logs/files/4 - METHYLATION & OS - (GPU) V1.txt'
SHUFFLE = True
RANDOM_STATE = 42  # if 'None' changes the seed to split training set and test set every time
LOWER_THRESHOLD = 1000  # 730 (2 years)
UPPER_THRESHOLD = 3000  # 2920 (8 years)
FIRST_FEATURES_SELECTION = 2000
SECOND_FEATURES_SELECTION = 200
VERBOSE = False
PLOT = False


## FUNCTIONS
def title(text):
    print(Fore.CYAN, '\n' + f' {text} '.center(80, '#'), Fore.RESET)


## MAIN
if __name__ == "__main__":
    # Open LOG file
    logfile = open(LOG_PATH, 'w')
    sys.stdout = DualOutput(sys.stdout, logfile)

    # Data Acquisition
    title('DATA ACQUISITION')
    dataset_paths = yaml_loader(DATASET_PATH_YAML)
    training_set, testing_set, dataset_columns = dataset_acquisition_and_splitting(
        path=dataset_paths[METHYLATION],
        json_paths_yaml=JSON_PATHS_YAML,
        names=METHYLATION_NAMES,
        shuffle=SHUFFLE,
        rand_state=RANDOM_STATE,
        lower_threshold=LOWER_THRESHOLD,
        upper_threshold=UPPER_THRESHOLD)

    # Training Set Management
    title('TRAINING SET MANAGEMENT')
    title('EXPLORATORY DATA ANALYSIS with RAW DATA')
    exploratory_data_analysis(dataframe=training_set, verbose=VERBOSE, plot=PLOT)
    title('FEATURES PREPROCESSING')
    dataset_columns.remove('y')
    training_set = features_preprocessing(
        dataframe=training_set,
        column_names=dataset_columns,
        verbose=VERBOSE)
    title('FEATURES SELECTION')
    training_set, selected_columns = features_selection_for_training(
        dataframe=training_set,
        variance_selection_dimension=FIRST_FEATURES_SELECTION,
        correlation_filter_dimension=SECOND_FEATURES_SELECTION)

    # Testing Set Management
    title('TESTING SET MANAGEMENT')
    title('EXPLORATORY DATA ANALYSIS with RAW DATA')
    exploratory_data_analysis(dataframe=testing_set, verbose=VERBOSE, plot=PLOT)
    title('FEATURES PREPROCESSING')
    testing_set = features_preprocessing(
        dataframe=testing_set,
        column_names=dataset_columns,
        verbose=VERBOSE)
    title('FEATURES SELECTION')
    testing_set = features_selection_for_testing(
        dataframe=testing_set,
        selected_features=selected_columns)

    # SKLearn to Torch
    title('SKLEARN TO TORCH')
    device, X_training_tensor, y_training_tensor, X_testing_tensor, y_testing_tensor = sklearn_to_torch(
        training_dataframe=training_set,
        testing_dataframe=testing_set)

    # Grid Search
    title('GRID SEARCH')
    best_parameters, sk_fold = grid_search(
        device=device,
        x=X_training_tensor,
        y=y_training_tensor,
        shuffle=SHUFFLE,
        rand_state=RANDOM_STATE,
        hyperparameters=general_hyperparameters(),
        k_folds=5,
        x_test=X_testing_tensor,
        y_test=y_testing_tensor)

    # Training
    title('TRAINING')
    model = training(
        device=device,
        x=X_training_tensor,
        y=y_training_tensor,
        shuffle=SHUFFLE,
        hyperparameters=best_parameters,
        sk_fold_setting=sk_fold)

    # Testing
    title('TESTING')
    testing(
        device=device,
        models=model,
        x_testing=X_testing_tensor,
        y_testing=y_testing_tensor)

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
