from colorama import Fore
from sklearn.model_selection import GridSearchCV


def grid_search(dataframe, catalogue, names, hyperparameters):
    """
        :param dataframe: preprocessed dataset loaded inside a dataframe
        :param catalogue: list of chosen models
        :param names: list of chosen models names
        :param hyperparameters: list of chosen models hyperparameters
        :return chosen_hyperparameters: list of optimal hyperparameters for each model
        :return trials: list of all the trials done with all the models
    """
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
