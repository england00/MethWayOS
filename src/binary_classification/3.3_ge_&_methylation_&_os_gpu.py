import os
from config.methods.configuration_loader import yaml_loader
from src.binary_classification.functions_sklearn.f1_dataset_acquisition import dataset_acquisition
from src.binary_classification.functions_sklearn.f2_exploratory_data_analysis import exploratory_data_analysis
from src.binary_classification.functions_sklearn.f3_features_preprocessing import features_preprocessing
from src.binary_classification.functions_sklearn.f4_features_selection import features_selection
from src.binary_classification.functions_sklearn.f5_dataset_splitting import dataset_splitting
from src.binary_classification.functions_torch.f6_sklearn_to_torch import sklearn_to_torch
from src.binary_classification.functions_torch.f7_hyperparameters import *
from src.binary_classification.functions_torch.f8_grid_search import grid_search
from src.binary_classification.functions_torch.f10_training_single_model import training
from src.binary_classification.functions_torch.f11_testing_single_model import testing
from logs.methods.log_storer import *


## CONFIGURATION
''' General '''
DATASET_PATH_YAML = '../../config/paths/dataset_paths.yaml'
LOG_PATH = f'../../logs/files/{os.path.basename(__file__)}.txt'
RANDOM_STATE = 42  # if 'None' changes the seed to split training set and test set every time
SHUFFLE = True
LOWER_THRESHOLD = 1000  # 730 (2 years)
UPPER_THRESHOLD = 3000  # 2920 (8 years)
PCA_DIMENSION = 80
PLOT = False
VERBOSE = False

''' Input Dataset '''
GENE_EXPRESSION_AND_METHYLATION_27 = 'gene_expression_and_methylation27'
GENE_EXPRESSION_AND_METHYLATION_450 = 'gene_expression_and_methylation450'  # only with 450 methylation island


## FUNCTIONS
def title(text):
    print('\n' + f' {text} '.center(80, '#'))


## MAIN
if __name__ == "__main__":
    # Open LOG file
    logfile = open(LOG_PATH, 'w')
    sys.stdout = DualOutput(sys.stdout, logfile)

    # Data Acquisition
    title('DATA ACQUISITION')
    dataset_paths = yaml_loader(DATASET_PATH_YAML)
    dataset, dataset_columns = dataset_acquisition(path=dataset_paths[GENE_EXPRESSION_AND_METHYLATION_27])

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

        # Features Selection
        title('FEATURES SELECTION')
        dataset_trial = features_selection(
            dataframe=dataset,
            rand_state=RANDOM_STATE,
            pca_dimension=PCA_DIMENSION,
            feature_number=features_number)

        # Dataset Splitting
        title('DATASET SPLITTING')
        training_set, testing_set = dataset_splitting(dataframe=dataset_trial, shuffle=SHUFFLE, rand_state=RANDOM_STATE)

        # SKLearn to Torch
        title('SKLEARN TO TORCH')
        device, X_training_tensor, y_training_tensor, X_testing_tensor, y_testing_tensor = sklearn_to_torch(
            training_dataframe=training_set,
            testing_dataframe=testing_set)

        # Grid Search
        title('GRID SEARCH')
        best_parameters, k_fold = grid_search(
            device=device,
            x=X_training_tensor,
            y=y_training_tensor,
            shuffle=SHUFFLE,
            rand_state=RANDOM_STATE,
            hyperparameters=double_layer_hyperparameters(),
            k_folds=5)

        # Training
        title('TRAINING')
        model = training(
            device=device,
            x=X_training_tensor,
            y=y_training_tensor,
            shuffle=SHUFFLE,
            hyperparameters=best_parameters,
            k_fold_setting=k_fold)

        # Testing
        title('TESTING')
        testing(
            device=device,
            model=model,
            x_testing=X_testing_tensor,
            y_testing=y_testing_tensor)

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
