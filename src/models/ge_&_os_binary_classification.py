import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from colorama import Fore
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from config.methods.configuration_loader import yaml_loader
from data.methods.csv_loader import csv_loader

## CONFIGURATION
JSON_PATHS_YAML = '../../config/files/json_paths.yaml'
DATASET_PATH_YAML = '../../config/files/dataset_paths.yaml'
GENE_EXPRESSION = 'gene_expression'
GENE_EXPRESSION_NAMES = 'gene_expression_names'
RANDOM_STATE = 42  # if 'None' changes the seed to split training set and test set every time
PLOT = False


## FUNCTIONS
def title(text):
    print(Fore.CYAN, '\n' + f' {text} '.center(80, '#'), Fore.RESET)


def data_acquisition(path):
    # Loading COLUMNS
    df, df_columns = csv_loader(path, JSON_PATHS_YAML, GENE_EXPRESSION_NAMES)

    # Splitting dataset in TRAINING and TESTING
    df_training, df_testing = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, shuffle=True)

    return df_training, df_testing, df_columns


def exploratory_data_analysis(dataframe, plot=True):
    # Showing some details about the dataset
    print('INFORMATION ABOUT THE TRAINING SET:')
    dataframe.info()
    print('\nDATA PREVIEW:\n', dataframe.head(3))

    # Plotting LABEL DISTRIBUTION
    if plot:
        sns.displot(dataframe['y'], color='green')
        plt.show()


def features_preprocessing(dataframe, lower_threshold=0, upper_threshold=0, test=False, column_names=None):
    # Operations to do ONLY with TRAINING SET
    if not test:

        # Checking possible values problems
        print('NAN VALUES:\n', dataframe.isnull().any())
        print('\nMISSING VALUES:\n', (dataframe == ' ').any())
        print('\nDUPLICATED VALUES:\n', dataframe.duplicated(keep='first'))

        # Managing imposed thresholds
        if lower_threshold != 0 and upper_threshold != 0:  # originally between 730 (2 years) and 2920 (8 years)
            i = j = 0
            for item in dataframe['y']:
                if item <= lower_threshold:
                    i += 1
                if item >= upper_threshold:
                    j += 1
            print('\nADMITTED SAMPLES VALUES:')
            print(f'- {i} with the label lower than {lower_threshold}')
            print(f'- {j} with the label bigger than {upper_threshold}')

    # Operations to do with BOTH TRAINING SET & TEST SET
    dataframe.loc[dataframe['y'] <= lower_threshold, 'y'] = 1  # changing lower values with '1'
    dataframe.loc[dataframe['y'] >= upper_threshold, 'y'] = 2  # changing higher values with '2'

    # Selecting only rows with labels '1' and '2'
    label_values = [1, 2]
    dataframe = dataframe[dataframe['y'].isin(label_values)]

    # Feature Scaling
    X = dataframe.drop('y', axis=1)
    y = dataframe['y']
    column_names.remove('y')
    scaler = StandardScaler().fit(X.astype('float64'))
    X = pd.DataFrame(scaler.transform(X[column_names].astype('float64')),
                     columns=column_names,
                     index=X[column_names].index)

    return pd.concat([X, y], axis=1, sort=False)


def models():
    catalogue = [DecisionTreeClassifier(class_weight='balanced'), SVC(class_weight='balanced')]
    names = ['Decision Tree', 'Support Vector Classifier']
    hyperparameters = [{'criterion': ['gini', 'entropy']},  # Decision Tree
                       {'C': [1e-4, 1e-2, 1, 1e1, 50, 1e2],  # Support Vector Classifier
                        'gamma': [0.005, 0.004, 0.003, 0.002, 0.001, 0.0005],
                        'kernel': ['linear', 'rbf']}]

    return catalogue, names, hyperparameters


def grid_search(catalogue, names, hyperparameters, dataset):
    chosen_hyperparameters = []
    trials = []

    # Searching the best hyperparameters set for each model
    for model, model_name, hyperparameter_set in zip(catalogue, names, hyperparameters):
        print('\n' + model_name.upper() + ':')
        clf = GridSearchCV(estimator=model, param_grid=hyperparameter_set, scoring='accuracy', cv=5)  # 'f1_weighted'
        clf.fit(dataset.drop('y', axis=1), dataset['y'])

        # Collecting chosen hyperparameters and estimators for ensembles
        chosen_hyperparameters.append(clf.best_params_)
        trials.append((model_name, clf))
        print('Accuracy:  ', Fore.GREEN, clf.best_score_, Fore.RESET)

        for hparam in hyperparameter_set:
            print(f'\t--> best value for hyperparameter "{hparam}": ',
                  Fore.YELLOW, clf.best_params_.get(hparam), Fore.RESET)

    return trials


## MAIN
if __name__ == "__main__":
    # Data Acquisition
    title('DATA ACQUISITION')
    dataset_paths = yaml_loader(DATASET_PATH_YAML)
    training_set, testing_set, set_columns = data_acquisition(path=dataset_paths[GENE_EXPRESSION])

    # Exploratory Data Analysis
    title('EXPLORATORY DATA ANALYSIS with RAW DATA')
    exploratory_data_analysis(dataframe=training_set, plot=PLOT)

    # FEATURES PREPROCESSING
    title('FEATURES PREPROCESSING')
    training_set = features_preprocessing(dataframe=training_set,
                                          lower_threshold=1000,
                                          upper_threshold=3000,
                                          column_names=set_columns)

    '''
    # REDUCING FEATURES
    # Supponiamo che X sia il tuo dataframe delle caratteristiche e y siano le etichette
    model = RandomForestClassifier(n_estimators=100)
    model.fit(training_set.drop('y', axis=1), training_set['y'])
    importances = model.feature_importances_
    selector = SelectFromModel(model, threshold='median', prefit=True)
    X_reduced = selector.transform(training_set.drop('y', axis=1))
    X_reduced_df = pd.DataFrame(X_reduced, columns=set_columns)

    training_set = pd.concat([X_reduced_df, training_set['y']], axis=1, sort=False)
    '''

    # MODELS SELECTION
    title('MODELS SELECTION')
    models_list, models_names, models_hyperparameters = models()

    # GRID SEARCH
    title('GRID SEARCH')
    estimators = grid_search(models_list, models_names, models_hyperparameters, training_set)
