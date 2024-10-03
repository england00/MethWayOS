from colorama import Fore
import numpy as np
import time
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier


def cross_validation_model_assessment(dataframe, hyperparameters, rand_state):
    """
        :param dataframe: training set loaded inside a dataframe
        :param hyperparameters: list of optimal hyperparameters for each model
        :param rand_state: chosen random seed
        :return models_dictionary: dictionary with all the model set with their best hyperparameters
    """
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
