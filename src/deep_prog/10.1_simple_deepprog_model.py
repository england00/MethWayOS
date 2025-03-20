from simdeep.simdeep_analysis import SimDeep
from simdeep.extract_data import LoadData
from simdeep.config import PATH_DATA
from simdeep.config import TRAINING_TSV
from simdeep.config import TEST_TSV
from simdeep.config import SURVIVAL_TSV
from simdeep.config import SURVIVAL_TSV_TEST


## CONFIGURATION
PATH_RESULTS = "./results/test_dummy/"


## MAIN
if __name__ == "__main__":
    ''' TRAINING '''
    # DATASET: instantiation
    dataset = LoadData(training_tsv=TRAINING_TSV,
            survival_tsv=SURVIVAL_TSV,
            path_data=PATH_DATA)

    # MODEL: instantiation with dummy example training dataset defined in the config file
    simDeep = SimDeep(
            dataset=dataset,
            path_results=PATH_RESULTS,
            path_to_save_model=PATH_RESULTS,
            )

    # MODEL: loading training dataset
    simDeep.load_training_dataset()

    # MODEL: fitting
    simDeep.fit()

    ''' TESTING '''
    # MODEL: loading testing dataset
    simDeep.load_new_test_dataset(
        TEST_TSV,
        fname_key='dummy',
        path_survival_file=SURVIVAL_TSV_TEST, # [OPTIONAL] test survival file useful to compute accuracy of test dataset
        )

    # The test set is a dummy rna expression (generated randomly)
    print(simDeep.dataset.test_tsv)
    # The data type of the test set is also defined to match an existing type
    print(simDeep.dataset.data_type)

    # MODEL: perform the classification analysis and label the set dataset
    simDeep.predict_labels_on_test_dataset()
    print(simDeep.test_labels)
    print(simDeep.test_labels_proba)

    # MODEL: saving
    simDeep.save_encoders('dummy_encoder.h5')
