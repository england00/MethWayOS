from colorama import Fore
from config.methods.configuration_loader import yaml_loader
from src.binary_classification.functions_sklearn.f1_dataset_acquisition import dataset_acquisition
from src.binary_classification.functions_sklearn.f2_exploratory_data_analysis import exploratory_data_analysis
from src.binary_classification.functions_sklearn.f3_features_preprocessing import features_preprocessing
from src.binary_classification.functions_sklearn.f4_features_selection import features_selection
from src.binary_classification.functions_sklearn.f5_dataset_splitting import dataset_splitting
from src.binary_classification.functions_sklearn.f6_models import models
from src.binary_classification.functions_sklearn.f7_grid_search import grid_search
from src.binary_classification.functions_sklearn.f8_cross_validation_model_assessment import cross_validation_model_assessment
from src.binary_classification.functions_sklearn.f9_training import training
from src.binary_classification.functions_sklearn.f10_testing import testing
from logs.methods.log_storer import *


## CONFIGURATION
JSON_PATHS_YAML = '../../config/files/json_paths.yaml'
DATASET_PATH_YAML = '../../config/files/dataset_paths.yaml'
GENE_EXPRESSION = 'gene_expression'
GENE_EXPRESSION_NAMES = 'gene_expression_names'
LOG_PATH = '../../logs/files/2 - GENE EXPRESSION & OS.txt'
SHUFFLE = False
RANDOM_STATE = None  # if 'None' changes the seed to split training set and test set every time
LOWER_THRESHOLD = 1000  # 730 (2 years)
UPPER_THRESHOLD = 3000  # 2920 (8 years)
PCA_DIMENSION = 90
FEATURES_NUMBER = 35
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
        path=dataset_paths[GENE_EXPRESSION],
        json_paths_yaml=JSON_PATHS_YAML,
        names=GENE_EXPRESSION_NAMES)

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

    # Features Selection
    title('FEATURES SELECTION')
    dataset = features_selection(
        dataframe=dataset,
        rand_state=RANDOM_STATE,
        pca_dimension=PCA_DIMENSION,
        feature_number=FEATURES_NUMBER)

    # Dataset Splitting
    title('DATASET SPLITTING')
    training_set, testing_set = dataset_splitting(dataframe=dataset, shuffle=SHUFFLE, rand_state=RANDOM_STATE)

    # Model Selection
    title('MODELS SELECTION')
    models_list, models_names, models_hyperparameters = models(rand_state=RANDOM_STATE)

    # Grid Search
    title('GRID SEARCH')
    best_parameters, estimators = grid_search(
        dataframe=training_set,
        catalogue=models_list,
        names=models_names,
        hyperparameters=models_hyperparameters)

    # Cross Validation
    title('CROSS VALIDATION')
    models_dictionary = cross_validation_model_assessment(
        dataframe=training_set,
        hyperparameters=best_parameters,
        rand_state=RANDOM_STATE)

    # Training
    title('TRAINING')
    final_models_dictionary = training(dataframe=training_set, models=models_dictionary)

    # Testing, Prediction & Metrics
    title('TESTING, PREDICTION & METRICS')
    testing(dataframe=testing_set, models=final_models_dictionary)

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
