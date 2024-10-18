from colorama import Fore
from config.methods.configuration_loader import yaml_loader
from src.binary_classification.functions_torch.f1_dataset_acquisition_and_splitting import dataset_acquisition_and_splitting
from src.binary_classification.functions_sklearn.f2_exploratory_data_analysis import exploratory_data_analysis
from src.binary_classification.functions_sklearn.f3_features_preprocessing import features_preprocessing
from logs.methods.log_storer import *


## CONFIGURATION
JSON_PATHS_YAML = '../../config/files/json_paths.yaml'
DATASET_PATH_YAML = '../../config/files/dataset_paths.yaml'
GENE_EXPRESSION = 'gene_expression'
GENE_EXPRESSION_NAMES = 'gene_expression_names'
LOG_PATH = '../../logs/files/4 - GENE EXPRESSION & OS - (GPU) V1.txt'
RANDOM_STATE = 42  # if 'None' changes the seed to split training set and test set every time
LOWER_THRESHOLD = 1000  # 730 (2 years)
UPPER_THRESHOLD = 3000  # 2920 (8 years)
PCA_DIMENSION = 80
VERBOSE = True
PLOT = True


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
        path=dataset_paths[GENE_EXPRESSION],
        json_paths_yaml=JSON_PATHS_YAML,
        names=GENE_EXPRESSION_NAMES,
        rand_state=RANDOM_STATE)

    # Exploratory Data Analysis
    title('EXPLORATORY DATA ANALYSIS with RAW DATA (TRAINING SET)')
    exploratory_data_analysis(dataframe=training_set, verbose=VERBOSE, plot=PLOT)

    # Feature Preprocessing
    title('FEATURES PREPROCESSING (TRAINING SET)')
    dataset_columns.remove('y')
    dataset = features_preprocessing(
        dataframe=training_set,
        lower_threshold=LOWER_THRESHOLD,
        upper_threshold=UPPER_THRESHOLD,
        verbose=VERBOSE,
        column_names=dataset_columns)





    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
