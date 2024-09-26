import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from colorama import Fore
from sklearn.ensemble import StackingClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from config.methods.configuration_loader import yaml_loader
from data.methods.csv_loader import csv_loader

## CONFIGURATION
JSON_PATHS_YAML = '../../config/files/json_paths.yaml'
DATASET_PATH_YAML = '../../config/files/dataset_paths.yaml'
GENE_EXPRESSION = 'gene_expression'
GENE_EXPRESSION_NAMES = 'gene_expression_names'
RANDOM_STATE = 42  # if 'None' changes the seed to split training set and test set every time
FEATURE_NUMBER = 36
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
    scaler = StandardScaler().fit(X.astype('float64'))
    X = pd.DataFrame(scaler.transform(X[column_names].astype('float64')),
                     columns=column_names,
                     index=X[column_names].index)

    return pd.concat([X, y], axis=1, sort=False)


def models():
    catalogue = [LogisticRegression(solver='saga', class_weight='balanced'),
                 KNeighborsClassifier(weights='distance'),
                 DecisionTreeClassifier(class_weight='balanced'),
                 SVC(class_weight='balanced')]
    names = ['Logistic Regression',
             'K-Nearest Neighbors',
             'Decision Tree',
             'Support Vector Classifier']
    hyperparameters = [{'penalty': ['l1', 'l2', 'elasticnet'],                          # Logistic Regression
                        'C': [1e-5,  1e-4, 1e-3, 1e-2, 0.01, 0.05, 0.07, 0.08, 0.09, 0.1, 0.5, 1],
                        'max_iter': [100, 300, 500]},
                       {'n_neighbors': list(range(1, 20, 2)),                           # K-Nearest Neighbors
                        'metric': ['euclidean', 'manhattan', 'minkowski'],
                        'p': [1, 2],  # valido solo per 'minkowski'
                        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
                       {'criterion': ['gini', 'entropy', 'log_loss'],                   # Decision Tree
                        'max_depth': [i for i in range(1, 30)],
                        'max_features': [None, 'sqrt', 'log2'],
                        'splitter': ['best', 'random']},
                       {'C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 40, 50, 60, 70, 1e2],     # Support Vector Classifier
                        'gamma': ['scale', 'auto', 0.005, 0.004, 0.003, 0.002, 0.001, 0.0005],
                        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}]

    return catalogue, names, hyperparameters


def grid_search(dataframe, catalogue, names, hyperparameters):
    chosen_hyperparameters = []
    trials = []

    # Searching the best hyperparameters set for each model
    for model, model_name, hyperparameter_set in zip(catalogue, names, hyperparameters):
        print('\n' + model_name.upper() + ':')
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

    return chosen_hyperparameters, trials


def cross_validation_model_assessment(dataframe, hyperparameters, trials):
    # Splitting FEATURES and LABELS
    X = dataframe.drop('y', axis=1)
    y = dataframe['y']

    # LOGISTIC REGRESSION
    logistic_regression_model = LogisticRegression(solver='saga',
                                                   class_weight='balanced',
                                                   C=hyperparameters[0]['C'],
                                                   max_iter=hyperparameters[0]['max_iter'],
                                                   penalty=hyperparameters[0]['penalty'],)
    scores = cross_validate(logistic_regression_model, X, y, cv=5, scoring=('f1_weighted', 'accuracy'), n_jobs=-1)
    print('\nLOGISTIC REGRESSION:')
    print(f'\t--> cross-validated Accuracy: ', Fore.GREEN, np.mean(scores['test_accuracy']), Fore.RESET)
    print(f'\t--> cross-validated We'
          f'ighted F1-score: ', Fore.GREEN, np.mean(scores['test_f1_weighted']), Fore.RESET)

    # K-NEAREST NEIGHBORS
    knn_model = KNeighborsClassifier(algorithm=hyperparameters[1]['algorithm'],
                                     metric=hyperparameters[1]['metric'],
                                     n_neighbors=hyperparameters[1]['n_neighbors'],
                                     p=hyperparameters[1]['p'])
    scores = cross_validate(knn_model, X, y, cv=5, scoring=('f1_weighted', 'accuracy'), n_jobs=-1)
    print('\nK-NEAREST NEIGHBORS:')
    print(f'\t--> cross-validated Accuracy: ', Fore.GREEN, np.mean(scores['test_accuracy']), Fore.RESET)
    print(f'\t--> cross-validated Weighted F1-score: ', Fore.GREEN, np.mean(scores['test_f1_weighted']), Fore.RESET)

    # Decision Tree
    decision_tree_model = DecisionTreeClassifier(class_weight='balanced',
                                                 criterion=hyperparameters[2]['criterion'],
                                                 max_depth=hyperparameters[2]['max_depth'],
                                                 max_features=hyperparameters[2]['max_features'],
                                                 splitter=hyperparameters[2]['splitter'])
    scores = cross_validate(decision_tree_model, X, y, cv=5, scoring=('f1_weighted', 'accuracy'), n_jobs=-1)
    print('\nDECISION TREE:')
    print(f'\t--> cross-validated Accuracy: ', Fore.GREEN, np.mean(scores['test_accuracy']), Fore.RESET)
    print(f'\t--> cross-validated Weighted F1-score: ', Fore.GREEN, np.mean(scores['test_f1_weighted']), Fore.RESET)

    # Support Vector Classifier
    support_vector_classifier_model = SVC(class_weight='balanced',
                                          C=hyperparameters[3]['C'],
                                          gamma=hyperparameters[3]['gamma'],
                                          kernel=hyperparameters[3]['kernel'])
    scores = cross_validate(support_vector_classifier_model, X, y, cv=5, scoring=('f1_weighted', 'accuracy'), n_jobs=-1)
    print('\nSUPPORT VECTOR CLASSIFIER:')
    print(f'\t--> cross-validated Accuracy: ', Fore.GREEN, np.mean(scores['test_accuracy']), Fore.RESET)
    print(f'\t--> cross-validated Weighted F1-score: ', Fore.GREEN, np.mean(scores['test_f1_weighted']), Fore.RESET)

    '''
    # STACKING CLASSIFIER with Logistic Regression, KNN and DT
    set_t0 = time.time()
    sf1_estimators = trials.copy()
    sf1_estimators.pop(3)  # removes Support Vector Classifier position from estimator list
    clf_stack1 = StackingClassifier(estimators=sf1_estimators, final_estimator=LogisticRegression())
    scores = cross_validate(clf_stack1, X, y, cv=5, scoring=('f1_weighted', 'accuracy'), n_jobs=-1)
    print('\nSTACKING CLASSIFIER with Logistic Regression, KNN and DT:')
    print(f'\t--> cross-validated Accuracy: ', Fore.GREEN, np.mean(scores['test_accuracy']), Fore.RESET)
    print(f'\t--> cross-validated Weighted F1-score: ', Fore.GREEN, np.mean(scores['test_f1_weighted']), Fore.RESET)
    print(f'\nValidation took {time.time() - set_t0} sec')

    # STACKING CLASSIFIER with Logistic Regression, KNN and SVC
    set_t0 = time.time()
    sf2_estimators = trials.copy()
    sf2_estimators.pop(2)  # removes Decision Tree position from estimator list
    clf_stack2 = StackingClassifier(estimators=sf2_estimators, final_estimator=LogisticRegression())
    scores = cross_validate(clf_stack2, X, y, cv=5, scoring=('f1_weighted', 'accuracy'), n_jobs=-1)
    print('\nSTACKING CLASSIFIER with Logistic Regression, KNN and SVC:')
    print(f'\t--> cross-validated Accuracy: ', Fore.GREEN, np.mean(scores['test_accuracy']), Fore.RESET)
    print(f'\t--> cross-validated Weighted F1-score: ', Fore.GREEN, np.mean(scores['test_f1_weighted']), Fore.RESET)
    print(f'\nValidation took {time.time() - set_t0} sec')
    '''

    # BAGGING CLASSIFIER with Decision Tree
    set_t0 = time.time()
    clf_bagging2 = BaggingClassifier(estimator=DecisionTreeClassifier(class_weight='balanced',
                                                                      criterion='entropy'),
                                     n_estimators=11)
    scores = cross_validate(clf_bagging2, X, y, cv=5, scoring=('f1_weighted', 'accuracy'), n_jobs=-1)
    print('\nRANDOM FOREST CLASSIFIER:')
    print(f'\t--> cross-validated Accuracy: ', Fore.GREEN, np.mean(scores['test_accuracy']), Fore.RESET)
    print(f'\t--> cross-validated Weighted F1-score: ', Fore.GREEN, np.mean(scores['test_f1_weighted']), Fore.RESET)
    print(f'\nValidation took {time.time() - set_t0} sec')

    # final model
    # return clf_stack1
    # return clf_stack2
    # return clf_bagging2


## MAIN
if __name__ == "__main__":
    # Data Acquisition
    title('DATA ACQUISITION')
    dataset_paths = yaml_loader(DATASET_PATH_YAML)
    training_set, testing_set, set_columns = data_acquisition(path=dataset_paths[GENE_EXPRESSION])

    # Exploratory Data Analysis
    title('EXPLORATORY DATA ANALYSIS with RAW DATA')
    exploratory_data_analysis(dataframe=training_set, plot=PLOT)

    # Feature Preprocessing
    title('FEATURES PREPROCESSING')
    set_columns.remove('y')
    training_set = features_preprocessing(dataframe=training_set,
                                          lower_threshold=1000,
                                          upper_threshold=3000,
                                          column_names=set_columns)

    # Reducing Features
    title('FEATURES PREPROCESSING')
    X_train = training_set.drop('y', axis=1)
    y_train = training_set['y']
    pca = PCA(n_components=FEATURE_NUMBER, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_train)
    df_pca = pd.DataFrame(X_pca, columns=[f'PC{i + 1}' for i in range(FEATURE_NUMBER)])
    df_pca['y'] = y_train.values

    # Model Selection
    title('MODELS SELECTION')
    models_list, models_names, models_hyperparameters = models()

    # Grid Search
    title('GRID SEARCH')
    best_parameters, estimators = grid_search(dataframe=df_pca,
                                              catalogue=models_list,
                                              names=models_names,
                                              hyperparameters=models_hyperparameters)

    # CROSS-VALIDATION
    title('CROSS-VALIDATION')
    cross_validation_model_assessment(dataframe=df_pca,
                                      hyperparameters=best_parameters,
                                      trials=estimators)
