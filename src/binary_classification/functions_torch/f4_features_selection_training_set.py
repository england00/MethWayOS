def features_selection_for_training(dataframe, variance_selection_dimension, correlation_filter_dimension):
    """
        :param dataframe: preprocessed dataset loaded inside a dataframe
        :param variance_selection_dimension: chosen VARIANCE selection dimension
        :param correlation_filter_dimension: chosen CORRELATION with label selection dimension
        :return dataframe_top_features: preprocessed dataset with only filtered features
        :return top_features: columns of the obtained dataframe
    """
    # Splitting dataset in FEATURE VECTORS and LABELS
    X = dataframe.drop('y', axis=1)
    y = dataframe['y']

    # Variance Selection
    print('VARIANCE SELECTION:')
    variance = X.var()
    top_features = variance.nlargest(variance_selection_dimension).index
    dataframe_top_features = X[top_features].copy()
    print(f"\t--> New Feature Space Dimension: {variance_selection_dimension}")

    # Correlation Filter
    print('CORRELATION FILTER:')
    correlations = dataframe_top_features.corrwith(y).abs()
    top_features = correlations.nlargest(correlation_filter_dimension).index
    dataframe_top_features = dataframe_top_features[top_features].copy()
    print(f"\t--> New Feature Space Dimension: {correlation_filter_dimension}")

    # Rebuilding Dataframe
    dataframe_top_features['y'] = y.values

    return dataframe_top_features, top_features
