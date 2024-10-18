def features_selection_for_testing(dataframe, selected_features):
    """
        :param dataframe: preprocessed dataset loaded inside a dataframe
        :param selected_features: columns of the dataframe to select
        :return dataframe_top_features: preprocessed dataset with only filtered features
    """
    # Splitting dataset in FEATURE VECTORS and LABELS
    X = dataframe.drop('y', axis=1)
    y = dataframe['y']

    # Top Features Selection
    print('TOP FEATURES SELECTION:')
    dataframe_top_features = X[selected_features].copy()
    print(f"\t--> New Feature Space Dimension: {len(selected_features)}")

    # Rebuilding Dataframe
    dataframe_top_features['y'] = y.values

    return dataframe_top_features
