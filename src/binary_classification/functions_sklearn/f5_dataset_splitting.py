from sklearn.model_selection import train_test_split


def dataset_splitting(dataframe, rand_state):
    """
        :param dataframe: preprocessed dataset loaded inside a dataframe
        :param rand_state: chosen random seed
        :return training_dataframe: training set loaded inside a dataframe
        :return testing_dataframe: testing set loaded inside a dataframe
    """
    # Splitting dataset in TRAINING and TESTING
    training_dataframe, testing_dataframe = train_test_split(dataframe,
                                                             test_size=0.2,
                                                             random_state=rand_state,
                                                             shuffle=True)
    print('DIMENSIONS:')
    print(f"\t--> Training Set: {len(training_dataframe)}")
    print(f"\t--> Testing Set: {len(testing_dataframe)}")

    return training_dataframe, testing_dataframe
