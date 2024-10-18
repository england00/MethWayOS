import matplotlib.pyplot as plt
import seaborn as sns


def exploratory_data_analysis(dataframe, verbose=True, plot=True):
    """
        :param dataframe: dataset loaded inside a dataframe
        :param verbose: printing selector for additional data on STDOUT
        :param plot: showing selector for GRAPHS
    """
    # Showing some details about the dataset
    if verbose:
        print('INFORMATION ABOUT THE DATASET:')
        dataframe.info()
        print('\nDATA PREVIEW:\n', dataframe.head(3))

    # Plotting LABEL DISTRIBUTION
    if plot:
        sns.displot(dataframe['y'], color='green')
        plt.show()
