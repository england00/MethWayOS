import pandas as pd
from sklearn.decomposition import PCA


def feature_selection(dataframe, rand_state, pca_dimension, feature_number):
    """
        :param dataframe: preprocessed dataset loaded inside a dataframe
        :param rand_state: chosen random seed
        :param pca_dimension: chosen PCA dimension
        :param feature_number: chosen feature_number
        :return selected_dataframe: preprocessed dataset with only filtered features
    """
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
