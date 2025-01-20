import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from config.methods.configuration_loader import *
from json_dir.methods.json_loader import *
from data.methods.csv_dataset_storer import *
from logs.methods.log_storer import *


## CONFIGURATION
''' General '''
DATASTORE_PATHS_YAML = '../../config/paths/datastore_paths.yaml'
DATASET_PATH_YAML = '../../config/paths/dataset_paths.yaml'
JSON_PATHS_YAML = '../../config/paths/json_paths.yaml'
LOG_PATH = f'../../logs/files/{os.path.basename(__file__)}.txt'
TABLE_PATHS_YAML = '../../config/paths/table_paths.yaml'

''' Input Datastore '''
GENE_ASSOCIATED_METHYLATION_27 = 'gene_associated_methylation_27'
GENE_ASSOCIATED_METHYLATION_450 = 'gene_associated_methylation_450'
METHYLATION_FULL_27 = 'methylation_full_27_MCAT'
METHYLATION_FULL_450 = 'methylation_full_450_MCAT'
OVERALL_SURVIVAL = 'overall_survival'

''' Output Keys '''
METHYLATION_27_KEYS = 'methylation27_keys'
METHYLATION_27_FULL_KEYS = 'methylation27_full_keys'
METHYLATION_450_KEYS = 'methylation450_keys'
METHYLATION_450_FULL_KEYS = 'methylation450_full_keys'

''' Output Dataset '''
METHYLATION_27_MCAT = 'methylation27_MCAT'
METHYLATION_27_FULL_MCAT = 'methylation27_full_MCAT'
METHYLATION_450_MCAT = 'methylation450_MCAT'  # only with 450k methylation island
METHYLATION_450_FULL_MCAT = 'methylation450_full_MCAT'


## FUNCTIONS
def process_patient(meth_patient, os_datastore, meth_keys):
    for patient in os_datastore:
        if meth_patient['info']['case_id'] == patient['info']['case_id']:
            # Case ID
            buffer = [meth_patient['info']['case_id']]
            # Survival Months
            if patient['last_check']['vital_status'] == 'Dead':  # DEAD cases
                buffer.append(round(float(patient['last_check']['days_to_death']) / 12, 2))  # Adding 'survival_months'
                buffer.append(0.0)  # Adding 'censorship'
            else:  # ALIVE cases
                buffer.append(
                    round(float(patient['last_check']['days_to_last_followup']) / 12, 2))  # Adding 'survival_months'
                buffer.append(1.0)  # Adding 'censorship'
            # Methylation Islands
            for key in meth_keys:
                buffer.append(meth_patient[key])  # Adding each feature
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
    table_paths = yaml_loader(TABLE_PATHS_YAML)

    # Storing data from JSON datastores
    methylation_datastore = json_loader(datastore_paths[METHYLATION_FULL_450])
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
    csv_storer(table_paths[METHYLATION_450_FULL_KEYS], methylation_keys, ["methylation_island"], keys_mode=True)
    gene_expression_keys = (['case_id', 'survival_months', 'censorship'] +
                            [key + '_meth' for key in methylation_keys])
    dataframe = pd.DataFrame(dataset, columns=gene_expression_keys)
    dataframe.to_csv(dataset_paths[METHYLATION_450_FULL_MCAT], index=True)
    print(f"Data has been correctly saved inside {dataset_paths[METHYLATION_450_FULL_MCAT]} file")

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
