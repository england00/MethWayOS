from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def models(rand_state):
    """
        :param rand_state: chosen random seed
        :return catalogue: list of chosen models
        :return names: list of chosen models names
        :return hyperparameters: list of chosen models hyperparameters
    """
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
                        'solver': ['adam']},
                       {'C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 40, 50, 60, 70, 1e2],  # Support Vector Classifier
                        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
                        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}]

    return catalogue, names, hyperparameters
