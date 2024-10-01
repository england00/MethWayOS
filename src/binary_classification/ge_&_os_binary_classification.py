import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from colorama import Fore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from config.methods.configuration_loader import yaml_loader
from data.methods.csv_loader import csv_loader
from logs.methods.log_storer import *


## CONFIGURATION
JSON_PATHS_YAML = '../../config/files/json_paths.yaml'
DATASET_PATH_YAML = '../../config/files/dataset_paths.yaml'
GENE_EXPRESSION = 'gene_expression'
GENE_EXPRESSION_NAMES = 'gene_expression_names'
LOG_PATH = '../../logs/files/2.3 - GENE EXPRESSION & OS - Binary Classification.txt'
RANDOM_STATE = None  # if 'None' changes the seed to split training set and test set every time
LOWER_THRESHOLD = 1000  # 730 (2 years)
UPPER_THRESHOLD = 3000  # 2920 (8 years)
PCA_DIMENSION = 90
FEATURES_NUMBER = 18
VERBOSE = False
PLOT = False


## FUNCTIONS
def title(text):
    print(Fore.CYAN, '\n' + f' {text} '.center(80, '#'), Fore.RESET)


def dataset_acquisition(path):
    # Loading COLUMNS
    dataframe, dataframe_columns = csv_loader(path, JSON_PATHS_YAML, GENE_EXPRESSION_NAMES)

    return dataframe, dataframe_columns


def exploratory_data_analysis(dataframe, verbose=True, plot=True):
    # Showing some details about the dataset
    if verbose:
        print('INFORMATION ABOUT THE DATASET:')
        dataframe.info()
        print('\nDATA PREVIEW:\n', dataframe.head(3))

    # Plotting LABEL DISTRIBUTION
    if plot:
        sns.displot(dataframe['y'], color='green')
        plt.show()


def features_preprocessing(dataframe, lower_threshold=0, upper_threshold=0, verbose=True, column_names=None):
    if verbose:
        # Checking possible values problems
        print('NAN VALUES:\n', dataframe.isnull().any())
        print('\nMISSING VALUES:\n', (dataframe == ' ').any())
        print('\nDUPLICATED VALUES:\n', dataframe.duplicated(keep='first'), '\n')

    # Managing imposed thresholds
    if lower_threshold != 0 and upper_threshold != 0:
        i = j = 0
        for item in dataframe['y']:
            if item <= lower_threshold:
                i += 1
            if item >= upper_threshold:
                j += 1
        print('ADMITTED SAMPLES VALUES:')
        print(f'\t--> {i} samples with a label lower than {lower_threshold}')
        print(f'\t--> {j} samples with a label bigger than {upper_threshold}')

    # Operations to do with BOTH TRAINING SET & TEST SET
    dataframe.loc[dataframe['y'] <= lower_threshold, 'y'] = 1  # changing lower values with '1'
    dataframe.loc[dataframe['y'] >= upper_threshold, 'y'] = 2  # changing higher values with '2'

    # Selecting only rows with labels '1' and '2'
    label_values = [1, 2]
    dataframe = dataframe[dataframe['y'].isin(label_values)]

    # Feature Scaling
    X = dataframe.drop('y', axis=1)
    y = dataframe['y']
    scaler = StandardScaler().fit(X.astype('float64'))
    X = pd.DataFrame(scaler.transform(X[column_names].astype('float64')),
                     columns=column_names,
                     index=X[column_names].index)

    return pd.concat([X, y], axis=1, sort=False)


def feature_selection(dataframe, rand_state, pca_dimension, feature_number):
    # Splitting dataset in FEATURE VECTORS and LABELS
    X = dataframe.drop('y', axis=1)
    y = dataframe['y']

    # Principal Component Analysis
    print('PRINCIPAL COMPONENT ANALYSIS:')
    pca = PCA(n_components=pca_dimension, random_state=rand_state)
    X_pca = pca.fit_transform(X)
    dataframe_pca = pd.DataFrame(X_pca, columns=[f'PC{i + 1}' for i in range(pca_dimension)])
    print(f"\t--> New Feature Space Dimension: {pca_dimension}")

    # Correlation Filter
    print('CORRELATION FILTER:')
    correlations = dataframe_pca.corrwith(y).abs()
    top_features = correlations.sort_values(ascending=False).head(feature_number).index
    selected_dataframe = dataframe_pca[top_features].copy()
    selected_dataframe['y'] = y.values
    print(f"\t--> New Feature Space Dimension: {feature_number}")

    return selected_dataframe


def dataset_splitting(dataframe, rand_state):
    # Splitting dataset in TRAINING and TESTING
    training_dataframe, testing_dataframe = train_test_split(dataframe,
                                                             test_size=0.2,
                                                             random_state=rand_state,
                                                             shuffle=True)
    print('DIMENSIONS:')
    print(f"\t--> Training Set: {len(training_dataframe)}")
    print(f"\t--> Testing Set: {len(testing_dataframe)}")

    return training_dataframe, testing_dataframe


def models(rand_state):
    catalogue = [DecisionTreeClassifier(class_weight='balanced'),
                 MLPClassifier(max_iter=10000, random_state=rand_state),
                 SVC(class_weight='balanced')]
    names = ['Decision Tree',
             'Multi-Layer Perceptron',
             'Support Vector Classifier']
    hyperparameters = [{'criterion': ['gini', 'entropy', 'log_loss'],  # Decision Tree
                        'max_depth': [i for i in range(1, 30)],
                        'max_features': [None, 'sqrt', 'log2'],
                        'min_samples_split': [2, 5, 10, 15],
                        'min_samples_leaf': [1, 2, 4, 6],
                        'splitter': ['best', 'random']},
                       {'alpha': [1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1],
                        'activation': ['logistic', 'tanh', 'relu'],
                        'hidden_layer_sizes': [(5,), (10,), (10, 5), (20,), (20, 10)],  # Multi-Layer Perceptron
                        'learning_rate': ['constant', 'adaptive'],
                        'learning_rate_init': [0.0001, 0.001, 0.01, 0.01],
                        'solver': ['adam', 'adamax']},
                       {'C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 40, 50, 60, 70, 1e2],  # Support Vector Classifier
                        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
                        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}]

    return catalogue, names, hyperparameters


def grid_search(dataframe, catalogue, names, hyperparameters):
    chosen_hyperparameters = []
    trials = []

    # Searching the best hyperparameters set for each model
    for model, model_name, hyperparameter_set in zip(catalogue, names, hyperparameters):
        print(model_name.upper() + ':')
        clf = GridSearchCV(estimator=model,
                           param_grid=hyperparameter_set,
                           scoring='accuracy',
                           cv=5,
                           n_jobs=-1)
        clf.fit(dataframe.drop('y', axis=1), dataframe['y'])

        # Collecting chosen hyperparameters and estimators for ensembles
        chosen_hyperparameters.append(clf.best_params_)
        trials.append((model_name, clf))
        print('Accuracy:  ', Fore.GREEN, clf.best_score_, Fore.RESET)

        for hparam in hyperparameter_set:
            print(f'\t--> best value for hyperparameter "{hparam}": ',
                  Fore.YELLOW, clf.best_params_.get(hparam), Fore.RESET)
        print('\n')

    return chosen_hyperparameters, trials


def cross_validation_model_assessment(dataframe, hyperparameters, rand_state):
    # Splitting FEATURES and LABELS
    X = dataframe.drop('y', axis=1)
    y = dataframe['y']

    # Decision Tree
    set_t0 = time.time()
    decision_tree_model = DecisionTreeClassifier(class_weight='balanced',
                                                 criterion=hyperparameters[0]['criterion'],
                                                 max_depth=hyperparameters[0]['max_depth'],
                                                 max_features=hyperparameters[0]['max_features'],
                                                 min_samples_split=hyperparameters[0]['min_samples_split'],
                                                 min_samples_leaf=hyperparameters[0]['min_samples_leaf'],
                                                 splitter=hyperparameters[0]['splitter'])
    scores = cross_validate(decision_tree_model, X, y, cv=5, scoring=('f1_weighted', 'accuracy'), n_jobs=-1)
    print('DECISION TREE:')
    print(f'\t--> cross-validated Accuracy: ', Fore.GREEN, np.mean(scores['test_accuracy']), Fore.RESET)
    print(f'\t--> cross-validated Weighted F1-score: ', Fore.GREEN, np.mean(scores['test_f1_weighted']), Fore.RESET)
    print(f'\t--> Validation took {time.time() - set_t0} sec')

    # Multi-Layer Perceptron
    set_t0 = time.time()
    mlp_model = MLPClassifier(max_iter=10000,
                              random_state=rand_state,
                              alpha=hyperparameters[1]['alpha'],
                              activation=hyperparameters[1]['activation'],
                              hidden_layer_sizes=hyperparameters[1]['hidden_layer_sizes'],
                              learning_rate=hyperparameters[1]['learning_rate'],
                              learning_rate_init=hyperparameters[1]['learning_rate_init'],
                              solver=hyperparameters[1]['solver'])
    scores = cross_validate(mlp_model, X, y, cv=5, scoring=('f1_weighted', 'accuracy'), n_jobs=-1)
    print('\nMULTI-LAYER PERCEPTRON:')
    print(f'\t--> cross-validated Accuracy: ', Fore.GREEN, np.mean(scores['test_accuracy']), Fore.RESET)
    print(f'\t--> cross-validated Weighted F1-score: ', Fore.GREEN, np.mean(scores['test_f1_weighted']), Fore.RESET)
    print(f'\t--> Validation took {time.time() - set_t0} sec')

    # Support Vector Classifier
    set_t0 = time.time()
    svc_model = SVC(class_weight='balanced',
                    C=hyperparameters[2]['C'],
                    gamma=hyperparameters[2]['gamma'],
                    kernel=hyperparameters[2]['kernel'])
    scores = cross_validate(svc_model, X, y, cv=5, scoring=('f1_weighted', 'accuracy'), n_jobs=-1)
    print('\nSUPPORT VECTOR CLASSIFIER:')
    print(f'\t--> cross-validated Accuracy: ', Fore.GREEN, np.mean(scores['test_accuracy']), Fore.RESET)
    print(f'\t--> cross-validated Weighted F1-score: ', Fore.GREEN, np.mean(scores['test_f1_weighted']), Fore.RESET)
    print(f'\t--> Validation took {time.time() - set_t0} sec')

    # BAGGING CLASSIFIER with Decision Tree
    set_t0 = time.time()
    ramdom_forest = BaggingClassifier(estimator=DecisionTreeClassifier(class_weight='balanced',
                                                                       criterion='gini'),
                                      n_estimators=11)
    scores = cross_validate(ramdom_forest, X, y, cv=5, scoring=('f1_weighted', 'accuracy'), n_jobs=-1)
    print('\nRANDOM FOREST CLASSIFIER:')
    print(f'\t--> cross-validated Accuracy: ', Fore.GREEN, np.mean(scores['test_accuracy']), Fore.RESET)
    print(f'\t--> cross-validated Weighted F1-score: ', Fore.GREEN, np.mean(scores['test_f1_weighted']), Fore.RESET)
    print(f'\t--> Validation took {time.time() - set_t0} sec')

    return {'Decision Tree': decision_tree_model,
            'Multi-Layer Perceptron': mlp_model,
            'Support Vector Classifier': svc_model,
            'Ramdom Forest': ramdom_forest}


def training(dataframe, model, name):
    # Splitting FEATURES and LABELS
    X = dataframe.drop('y', axis=1)
    y = dataframe['y']

    # Training the model
    set_t0 = time.time()
    model.fit(X, y)
    print(f'{name.upper()}\' training took {time.time() - set_t0} sec')

    return model


def testing(dataframe, models):
    # Splitting FEATURES and LABELS
    X = dataframe.drop('y', axis=1)
    y = dataframe['y']

    # Testing each model with Testing Set
    y_pred = {}
    for item in models:
        set_t0 = time.time()
        y_pred[item] = models[item].predict(X)
        print(f'{item.upper()}\' testing took {time.time() - set_t0} sec')

    return X, y, y_pred


def results(y, y_pred, name):
    # metrics
    print(f'{name.upper()}:')
    print('\t--> Accuracy: ', accuracy_score(y, y_pred))
    print('\t--> Precision: ', precision_score(y, y_pred, average='weighted'))
    print('\t--> Recall: ', recall_score(y, y_pred, average='weighted'))
    print('\t--> F1-Score: ', f1_score(y, y_pred, average='weighted'))


## MAIN
if __name__ == "__main__":

    # Open LOG file
    logfile = open(LOG_PATH, 'w')
    sys.stdout = DualOutput(sys.stdout, logfile)

    # Data Acquisition
    title('DATA ACQUISITION')
    dataset_paths = yaml_loader(DATASET_PATH_YAML)
    dataset, dataset_columns = dataset_acquisition(path=dataset_paths[GENE_EXPRESSION])

    # Exploratory Data Analysis
    title('EXPLORATORY DATA ANALYSIS with RAW DATA')
    exploratory_data_analysis(dataframe=dataset, verbose=VERBOSE, plot=PLOT)

    # Feature Preprocessing
    title('FEATURES PREPROCESSING')
    dataset_columns.remove('y')
    dataset = features_preprocessing(dataframe=dataset,
                                     lower_threshold=LOWER_THRESHOLD,
                                     upper_threshold=UPPER_THRESHOLD,
                                     verbose=VERBOSE,
                                     column_names=dataset_columns)

    # Feature Selection
    title('FEATURE SELECTION')
    dataset = feature_selection(dataframe=dataset,
                                rand_state=RANDOM_STATE,
                                pca_dimension=PCA_DIMENSION,
                                feature_number=FEATURES_NUMBER)

    # Dataset Splitting
    title('DATASET SPLITTING')
    training_set, testing_set = dataset_splitting(dataframe=dataset, rand_state=RANDOM_STATE)

    # Model Selection
    title('MODELS SELECTION')
    models_list, models_names, models_hyperparameters = models(rand_state=RANDOM_STATE)

    # Grid Search
    title('GRID SEARCH')
    best_parameters, estimators = grid_search(dataframe=training_set,
                                              catalogue=models_list,
                                              names=models_names,
                                              hyperparameters=models_hyperparameters)

    # Cross Validation
    title('CROSS VALIDATION')
    models_dictionary = cross_validation_model_assessment(dataframe=training_set,
                                                          hyperparameters=best_parameters,
                                                          rand_state=RANDOM_STATE)

    # Training
    title('TRAINING')
    final_models_dictionary = {}
    for item in models_dictionary:
        final_models_dictionary[item] = training(dataframe=training_set,
                                                 model=models_dictionary[item],
                                                 name=item)

    # Testing & Prediction
    title('TESTING AND PREDICTION')
    X_testing, y_testing, y_prediction = testing(dataframe=testing_set,
                                                 models=final_models_dictionary)

    # Final Testing Results
    title('FINAL TESTING RESULTS')
    for item in final_models_dictionary:
        results(y=y_testing,
                y_pred=y_prediction[item],
                name=item)

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
