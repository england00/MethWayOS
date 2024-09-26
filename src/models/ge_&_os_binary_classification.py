import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from colorama import Fore
from sklearn.model_selection import train_test_split
from config.methods.configuration_loader import yaml_loader
from data.methods.csv_loader import csv_loader


## CONFIGURATION
JSON_PATHS_YAML = '../../config/files/json_paths.yaml'
DATASET_PATH_YAML = '../../config/files/dataset_paths.yaml'
GENE_EXPRESSION = 'gene_expression'
GENE_EXPRESSION_NAMES = 'gene_expression_names'


## FUNCTIONS
def title(text):
    print(Fore.CYAN, '\n' + f' {text} '.center(80, '#'), Fore.RESET)


## LOADING DATASET
def loading_dataset(path):
    # Loading COLUMNS
    df, df_columns = csv_loader(path, JSON_PATHS_YAML, GENE_EXPRESSION_NAMES)

    # Splitting dataset in TRAINING and TESTING
    df_training, df_testing = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    return df_training, df_testing, df_columns


## MAIN
if __name__ == "__main__":
    # Data Acquisition
    title('DATA ACQUISITION')
    dataset_paths = yaml_loader(DATASET_PATH_YAML)
    training_set, testing_set, set_columns = loading_dataset(dataset_paths[GENE_EXPRESSION])

