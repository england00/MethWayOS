import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier, BaggingClassifier
from sklearn.feature_selection import SequentialFeatureSelector, VarianceThreshold, SelectKBest, f_regression, \
    SelectFromModel
import time
from colorama import Fore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
from scikitplot.metrics import plot_roc_curve, plot_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from lifelines.utils import concordance_index

from config.methods.configuration_loader import *
from json_dir.methods.json_loader import *


## CONFIGURATION
JSON_PATHS_YAML = '../../config/files/json_paths.yaml'
DATASET_PATH_YAML = '../../config/files/dataset_paths.yaml'
GENE_EXPRESSION = 'gene_expression'
GENE_EXPRESSION_NAMES = 'gene_expression_names'


def title(text):
    print(Fore.CYAN, '\n' + f' {text} '.center(80, '#'), Fore.RESET)


## LOADING DATASET
def loading_dataset(path):
    # Loading COLUMNS
    json_paths = yaml_loader(JSON_PATHS_YAML)
    names = json_loader(json_paths[GENE_EXPRESSION_NAMES])
    names.append('y')
    df = pd.read_csv(path, delimiter=';', header=None, names=names)

    # Splitting dataset in TRAINING and TESTING
    df_training, df_testing = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    return df_training, df_testing, names




## MAIN
if __name__ == "__main__":
    # Data Acquisition
    title('DATA ACQUISITION')
    dataset_paths = yaml_loader(DATASET_PATH_YAML)
    training_set, testing_set, features_names = loading_dataset(dataset_paths[GENE_EXPRESSION])



    scaler = StandardScaler()  # Oppure MinMaxScaler()

    variance_threshold = VarianceThreshold(threshold=0.01)  # Soglia da regolare

    k_best = 500  # Numero di feature da mantenere, da regolare
    select_kbest = SelectKBest(score_func=f_regression, k=k_best)

    lasso = SelectFromModel(Lasso(alpha=0.001, max_iter=10000))

    pca = PCA(n_components=70, random_state=42)  # Numero di componenti da regolare


    pipeline = Pipeline([
        ('scaling', scaler),
        ('variance_threshold', variance_threshold),
        ('select_kbest', select_kbest),
        ('lasso', lasso),
        ('pca', pca)
    ])

    X_train = training_set.drop('y', axis=1)
    y_train = training_set['y']
    X_test = testing_set.drop('y', axis=1)
    y_test = testing_set['y']

    pipeline.fit(X_train, y_train)

    X_train_transformed = pipeline.transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    title('TRAINING')
    model = LinearRegression()
    model.fit(X_train_transformed, y_train)

    title('TEST')
    y_pred = model.predict(X_test_transformed)

    title('C-INDEX')


    c_index = concordance_index(y_test, y_pred)
    print(f"C-index: {c_index:.4f}")

    score = r2_score(y_test, y_pred)
    print(f"R² sul test set: {score:.4f}")
