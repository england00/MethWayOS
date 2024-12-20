import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from config.methods.configuration_loader import *
from json_dir.methods.json_loader import *
from data.methods.csv_dataset_storer import *
from logs.methods.log_storer import *


## CONFIGURATION
DATASTORE_PATHS_YAML = '../../config/paths/datastore_paths.yaml'
DATASET_PATH_YAML = '../../config/paths/dataset_paths.yaml'
GENE_ASSOCIATED_METHYLATION = 'gene_associated_methylation'
JSON_PATHS_YAML = '../../config/paths/json_paths.yaml'
LOG_PATH = f'../../logs/files/{os.path.basename(__file__)}.txt'
METHYLATION_27 = 'methylation27'
METHYLATION_450 = 'methylation450'  # only with 450 methylation island
OVERALL_SURVIVAL = 'overall_survival'


## FUNCTIONS
def process_patient(meth_patient, os_datastore, meth_keys):
    for patient in os_datastore:
        if meth_patient['info']['case_id'] == patient['info']['case_id']:
            buffer = []
            for key in meth_keys:
                buffer.append(meth_patient[key])  # Adding each feature
            if patient['last_check']['vital_status'] == 'Dead':  # DEAD cases
                buffer.append(patient['last_check']['days_to_death'])  # Adding label
            else:  # ALIVE cases
                buffer.append(patient['last_check']['days_to_last_followup'])  # Adding label
            return buffer
    return None


## MAIN
if __name__ == "__main__":

    # Open LOG file
    logfile = open(LOG_PATH, 'w')
    sys.stdout = DualOutput(sys.stdout, logfile)

    # Loading YAML paths
    json_paths = yaml_loader(JSON_PATHS_YAML)
    datastore_paths = yaml_loader(DATASTORE_PATHS_YAML)
    dataset_paths = yaml_loader(DATASET_PATH_YAML)

    # Storing data from JSON datastores
    methylation_datastore = json_loader(datastore_paths[GENE_ASSOCIATED_METHYLATION])
    overall_survival_datastore = json_loader(datastore_paths[OVERALL_SURVIVAL])

    # Creating the dataset with METHYLATION as feature vector and OVERALL SURVIVAL as label
    methylation_keys = [key for key in methylation_datastore[0].keys() if key != 'info']
    num_patients = len(methylation_datastore)
    dataset = []

    # Using ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        futures = []
        with tqdm(total=num_patients, desc="Processing cases") as pbar:  # Progress bar
            for methylation_patient in methylation_datastore:
                futures.append(executor.submit(process_patient,
                                               methylation_patient,
                                               overall_survival_datastore,
                                               methylation_keys))

            # Process each future as it completes
            for future in as_completed(futures):
                result = future.result()
                if result:
                    dataset.append(result)
                pbar.update(1)  # Update the progress bar for each completed task
    print(f"Loaded {len(dataset)} samples")

    # Storing dataset inside a CSV file
    csv_storer(dataset_paths[METHYLATION_27], dataset, methylation_keys)

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
