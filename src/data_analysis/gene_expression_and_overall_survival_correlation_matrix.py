import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier, BaggingClassifier
from sklearn.feature_selection import SequentialFeatureSelector
import time
from colorama import Fore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scikitplot.metrics import plot_roc_curve, plot_confusion_matrix

from config.methods.configuration_loader import *
from json_dir.methods.json_loader import *

## CONFIGURATION
JSON_PATHS_YAML = '../../config/files/json_paths.yaml'
DATASET_PATH_YAML = '../../config/files/dataset_paths.yaml'
GENE_EXPRESSION = 'gene_expression'
GENE_EXPRESSION_NAMES = 'gene_expression_names'

PLOT = True

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

pd.set_option('display.max_columns', None)
plt.rcParams.update({'figure.max_open_warning': 0})


def title(text):
    print(Fore.CYAN, '\n' + f' {text} '.center(60, '#'), Fore.RESET)


############################################### LOADING DATASET ########################################################

def loading_dataset(path):
    # Loading COLUMNS
    json_paths = yaml_loader(JSON_PATHS_YAML)
    names = json_loader(json_paths[GENE_EXPRESSION_NAMES])
    names.append('y')
    df = pd.read_csv(path, delimiter=';', header=None, names=names)

    # selecting features and target variable
    y = df['y']
    X = df.drop(columns='y')

    # plotting the correlation matrix
    correlation_df = pd.concat([X, y], axis=1, sort=False)
    correlation_matrix = correlation_df.corr()
    correlation_with_y = correlation_matrix['y'].drop('y')
    selected_features = correlation_with_y[abs(correlation_with_y) > 0.3]

    print(selected_features)
    # X_selected = X[selected_features]

    # splitting the dataset in training and testing

    # df_tr, df_ts = train_test_split(df, test_size=0.2, random_state=42, shuffle=True, stratify=df['y'])

    # return df_tr, df_ts, names


## MAIN
if __name__ == "__main__":
    # Data Acquisition
    title('DATA ACQUISITION')
    dataset_paths = yaml_loader(DATASET_PATH_YAML)
    loading_dataset(dataset_paths[GENE_EXPRESSION])

    # training_set, testing_set, features_names = loading_dataset(dataset_paths[GENE_EXPRESSION])
