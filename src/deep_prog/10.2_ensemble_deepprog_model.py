from collections import OrderedDict
import ray
from simdeep.config import PATH_DATA
from simdeep.simdeep_boosting import SimDeepBoosting


## CONFIGURATION
TSV_FILES = OrderedDict([
          ('METH', 'meth_dummy.tsv'),
          ('RNA', 'rna_dummy.tsv'),
])
SURVIVAL_TSV = 'survival_dummy.tsv'
PROJECT_NAME = 'test_dummy_stacked'
EPOCHS = 10     # Autoencoder epochs. Other hyperparameters can be fine-tuned
SEED = 3        # Random seed used for reproducibility
NB_ITERATION = 5       # Number of models to be fitted using only a subset of the training data
NB_THREADS = 2  # These treads define the number of threads to be used to compute survival function
PATH_RESULTS = "./results/"


## MAIN
if __name__ == "__main__":
    # DISTRIBUTED COMPUTING: instantiation
    ray.init(num_cpus=3)

    ''' TRAINING '''
    # MODEL: instantiation with dummy example training dataset defined in the config file
    boosting = SimDeepBoosting(
        nb_threads=NB_THREADS,
        nb_it=NB_ITERATION,
        split_n_fold=3,
        survival_tsv=SURVIVAL_TSV,
        training_tsv=TSV_FILES,
        path_data=PATH_DATA,
        project_name=PROJECT_NAME,
        path_results=PATH_RESULTS,
        epochs=EPOCHS,
        seed=SEED,
        distribute=True
    )

    # MODEL: fitting
    boosting.fit()

    # MODEL: predicting and writing the labels
    boosting.predict_labels_on_full_dataset()

    # MODEL: computing and writing the feature importance
    boosting.compute_feature_scores_per_cluster()
    boosting.write_feature_score_per_cluster()

    # MODEL: computing internal metrics
    boosting.compute_clusters_consistency_for_full_labels()
    boosting.compute_c_indexes_for_full_dataset()

    # MODEL: evaluating cluster performance
    boosting.evalutate_cluster_performance()

    # MODEL: collecting more c-indexes
    boosting.collect_cindex_for_test_fold()
    boosting.collect_cindex_for_full_dataset()
    boosting.collect_cindex_for_training_dataset()

    # MODEL: seeing average number of significant features per omic across OMIC models
    boosting.collect_number_of_features_per_omic()

    ''' TESTING '''
    # MODEL: loading testing dataset
    boosting.load_new_test_dataset(
        {'RNA': 'rna_dummy.tsv'},  # OMIC file of the test set. It doesnt have to be the same as for training
        'TEST_DATA_1',  # Name of the test test to be used
        'survival_dummy.tsv',
        # [OPTIONAL] Survival file of the test set. USeful to compute accuracy metrics on the test dataset
    )

    # MODEL: predicting labels on the test dataset
    boosting.predict_labels_on_test_dataset()

    # MODEL: computing C-index
    boosting.compute_c_indexes_for_test_dataset()

    # MODEL: seeing cluster consistency
    boosting.compute_clusters_consistency_for_test_labels()

    # MODEL: visualization
    boosting.plot_supervised_kernel_for_test_sets()
    boosting.plot_supervised_predicted_labels_for_test_sets()

    # DISTRIBUTED COMPUTING: shutdown
    ray.shutdown()
