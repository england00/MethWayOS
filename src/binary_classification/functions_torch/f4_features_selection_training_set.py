def features_selection_for_training(dataframe,
                                    variance_selection_dimension,
                                    intermediate_correlation_threshold,
                                    correlation_filter_dimension):
    """
        :param dataframe: preprocessed dataset loaded inside a dataframe
        :param variance_selection_dimension: chosen VARIANCE selection dimension
        :param intermediate_correlation_threshold: chosen maximum CORRELATION threshold among the features
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

    # Intermediate Correlation Filtering
    print('CORRELATION FILTER: removing most correlated features with each other')
    corr_matrix = dataframe_top_features.corr().abs()
    features_to_drop = []
    for i in range(corr_matrix.shape[0]):
        for j in range(i + 1, corr_matrix.shape[1]):
            if corr_matrix.iloc[i, j] > intermediate_correlation_threshold:
                # Choosing features with higher variability
                if variance[top_features[i]] < variance[top_features[j]]:
                    features_to_drop.append(top_features[i])
                else:
                    features_to_drop.append(top_features[j])
    dataframe_top_features.drop(columns=set(features_to_drop), inplace=True)
    print(f"\t--> New Feature Space Dimension after Intermediate Correlation Filter: {dataframe_top_features.shape[1]}")

    # Correlation Filter
    print('CORRELATION FILTER: removing least correlated features with label')
    correlations = dataframe_top_features.corrwith(y).abs()
    top_features = correlations.nlargest(correlation_filter_dimension).index
    dataframe_top_features = dataframe_top_features[top_features].copy()
    print(f"\t--> New Feature Space Dimension: {correlation_filter_dimension}")

    # Rebuilding Dataframe
    dataframe_top_features['y'] = y.values

    return dataframe_top_features, top_features
