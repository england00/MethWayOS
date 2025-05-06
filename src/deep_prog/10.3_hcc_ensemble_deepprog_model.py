from collections import OrderedDict
from os.path import isfile
import ray
from simdeep.simdeep_boosting import SimDeepBoosting


## CONFIGURATION
PATH_DATA = '../../data/datasets/'
assert(isfile(PATH_DATA + "/KIRCKIRP_DeepProg_methylation.tsv"))
assert(isfile(PATH_DATA + "/KIRCKIRP_DeepProg_gene_expression.tsv"))
TSV_FILES = OrderedDict([
    ('METH', 'KIRCKIRP_DeepProg_methylation.tsv'),
    ('RNA', 'KIRCKIRP_DeepProg_gene_expression.tsv'),
])
assert(isfile(PATH_DATA + "KIRCKIRP_DeepProg_overall_survival.tsv"))
SURVIVAL_TSV = 'KIRCKIRP_DeepProg_overall_survival.tsv'
PROJECT_NAME = 'test_KIRCKIRP_stacked'
EPOCHS = 25             # Autoencoder epochs. Other hyperparameters can be fine-tuned
SEED = 10045            # Random seed used for reproducibility
NB_ITERATION = 9       # Number of models to be fitted using only a subset of the training data
NB_THREADS = 2          # These treads define the number of threads to be used to compute survival function
PATH_RESULTS = "./results/"
SURVIVAL_FLAG = {
    'patient_id': 'Samples',
    'survival': 'days',
    'event': 'event'
}


## MAIN
if __name__ == "__main__":
    # DISTRIBUTED COMPUTING: instantiation
    ray.init(num_cpus=1)

    ''' TRAINING '''
    # MODEL: instantiation with dummy example training dataset defined in the config file
    boosting = SimDeepBoosting(
        nb_threads=NB_THREADS,
        nb_it=NB_ITERATION,
        split_n_fold=5,
        survival_tsv=SURVIVAL_TSV,
        training_tsv=TSV_FILES,
        path_data=PATH_DATA,
        project_name=PROJECT_NAME,
        path_results=PATH_RESULTS,
        epochs=EPOCHS,
        survival_flag=SURVIVAL_FLAG,
        distribute=True,
        seed=SEED)

    # MODEL: fitting
    boosting.fit()

    # MODEL: predicting and writing the labels
    boosting.predict_labels_on_full_dataset()

    # MODEL: computing internal metrics
    boosting.compute_clusters_consistency_for_full_labels()
    boosting.evalutate_cluster_performance()
    boosting.collect_cindex_for_test_fold()
    boosting.collect_cindex_for_full_dataset()

    # MODEL: computing and writing the feature importance
    boosting.compute_feature_scores_per_cluster()
    boosting.write_feature_score_per_cluster()

    ''' TESTING '''
    # MODEL: loading testing dataset
    boosting.load_new_test_dataset(
        {'RNA': 'KIRCKIRP_DeepProg_gene_expression.tsv'},
        'test_RNA_only',
        SURVIVAL_TSV,
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
