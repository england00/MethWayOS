from colorama import Fore
from config.methods.configuration_loader import yaml_loader
from src.binary_classification.functions.f1_dataset_acquisition import dataset_acquisition
from src.binary_classification.functions.f2_exploratory_data_analysis import exploratory_data_analysis
from src.binary_classification.functions.f3_features_preprocessing import features_preprocessing
from src.binary_classification.functions.f4_feature_selection import feature_selection
from src.binary_classification.functions.f5_dataset_splitting import dataset_splitting
from src.torch_tests.functions.f6_sklearn_to_torch import sklearn_to_torch
from src.torch_tests.functions.f7_hyperparameters import *
from src.torch_tests.functions.f8_grid_search import grid_search
from src.torch_tests.functions.f10_training import training
from src.torch_tests.functions.f11_testing import testing
from logs.methods.log_storer import *


## CONFIGURATION
JSON_PATHS_YAML = '../../config/files/json_paths.yaml'
DATASET_PATH_YAML = '../../config/files/dataset_paths.yaml'
GENE_EXPRESSION_AND_METHYLATION = 'gene_expression_and_methylation'
GENE_EXPRESSION_AND_METHYLATION_NAMES = 'gene_expression_and_methylation_names'
LOG_PATH = '../../logs/files/4.4 - GENE EXPRESSION & METHYLATION & OS - Binary Classification (GPU).txt'
RANDOM_STATE = 42  # if 'None' changes the seed to split training set and test set every time
LOWER_THRESHOLD = 1000  # 730 (2 years)
UPPER_THRESHOLD = 3000  # 2920 (8 years)
PCA_DIMENSION = 80
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
    dataset, dataset_columns = dataset_acquisition(
        path=dataset_paths[GENE_EXPRESSION_AND_METHYLATION],
        json_paths_yaml=JSON_PATHS_YAML,
        names=GENE_EXPRESSION_AND_METHYLATION_NAMES)

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

    # TRIALS
    for features_number in range(8, 40):
        title(f'TRIAL: {features_number} features')

        # Feature Selection
        title('FEATURE SELECTION')
        dataset_trial = feature_selection(
            dataframe=dataset,
            rand_state=RANDOM_STATE,
            pca_dimension=PCA_DIMENSION,
            feature_number=features_number)

        # Dataset Splitting
        title('DATASET SPLITTING')
        training_set, testing_set = dataset_splitting(dataframe=dataset_trial, rand_state=RANDOM_STATE)

        # SKLearn to Torch
        title('SKLEARN TO TORCH')
        device, X_training_tensor, y_training_tensor, X_testing_tensor, y_testing_tensor = sklearn_to_torch(
            training_dataframe=training_set,
            testing_dataframe=testing_set)

        # Hyperparameters Lists
        title('HYPERPARAMETERS LISTS')
        hyperparameters = double_layer_hyperparameters()

        # Grid Search
        title('GRID SEARCH')
        best_parameters = grid_search(
            device=device,
            X=X_training_tensor,
            y=y_training_tensor,
            rand_state=RANDOM_STATE,
            hyperparameters=hyperparameters)

        # Training
        title('TRAINING')
        model = training(
            device=device,
            X=X_training_tensor,
            y=y_training_tensor,
            X_validation=X_testing_tensor,
            y_validation=y_testing_tensor,
            hyperparameters=best_parameters)

        # Testing
        title('TESTING')
        testing(
            model=model,
            X_testing=X_testing_tensor,
            y_testing=y_testing_tensor)

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
