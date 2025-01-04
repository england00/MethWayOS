import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from config.methods.configuration_loader import *
from json_dir.methods.json_loader import *
from data.methods.csv_dataset_storer import *
from logs.methods.log_storer import *


## CONFIGURATION
DATASET_PATH_YAML = '../../config/paths/dataset_paths.yaml'
DATASTORE_PATHS_YAML = '../../config/paths/datastore_paths.yaml'
GENE_EXPRESSION_MCAT = 'gene_expression_MCAT'
GENE_EXPRESSION_KEYS = 'gene_expression_keys'
JSON_PATHS_YAML = '../../config/paths/json_paths.yaml'
LOG_PATH = f'../../logs/files/{os.path.basename(__file__)}.txt'
OVERALL_SURVIVAL = 'overall_survival'
TABLE_PATHS_YAML = '../../config/paths/table_paths.yaml'
WHOLE_SLIDE_IMAGE = 'whole_slide_image'


## FUNCTIONS
def process_patient(ge_patient, os_datastore, wsi_datastore, ge_keys):
    for patient in os_datastore:
        if (ge_patient['info']['case_id'] in wsi_datastore and
                ge_patient['info']['case_id'] == patient['info']['case_id']):
            # Case ID & Slide ID & Patient ID
            buffer = [ge_patient['info']['case_id'],
                      wsi_datastore[patient['info']['case_id']]['slide_id'],
                      wsi_datastore[patient['info']['case_id']]['patient']]
            # Survival Months
            if patient['last_check']['vital_status'] == 'Dead':  # DEAD cases
                buffer.append(round(float(patient['last_check']['days_to_death']) / 12, 2))  # Adding 'survival_months'
                buffer.append(0.0)  # Adding 'censorship'
            else:  # ALIVE cases
                buffer.append(round(float(patient['last_check']['days_to_last_followup']) / 12, 2))  # Adding 'survival_months'
                buffer.append(1.0)  # Adding 'censorship'
            # Gene Expression RNASeq
            for key in ge_keys:
                buffer.append(ge_patient[key][7])  # Adding each '_rnaseq' gene
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
    gene_expression_datastore = json_loader(datastore_paths[GENE_EXPRESSION_MCAT])
    overall_survival_datastore = json_loader(datastore_paths[OVERALL_SURVIVAL])
    whole_slide_image_datastore = json_loader(datastore_paths[WHOLE_SLIDE_IMAGE])

    # Changing 'gene_id' with 'gene_name' as key in each dictionary
    for gene_expression_patient in gene_expression_datastore:
        new_item = {'info': gene_expression_patient['info']}
        for gene_key, value in gene_expression_patient.items():
            if gene_key != 'info':
                new_key = value[0]
                new_item[new_key] = [gene_key] + value[1:]
        gene_expression_patient.clear()
        gene_expression_patient.update(new_item)

    # Creating the dataset with GENE EXPRESSION as feature vector and OVERALL SURVIVAL as label
    gene_expression_keys = [key for key in gene_expression_datastore[0].keys() if key != 'info']
    num_patients = len(gene_expression_datastore)
    dataset = []

    # Using ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        futures = []
        with tqdm(total=num_patients, desc="Processing cases") as pbar:  # Progress bar
            for gene_expression_patient in gene_expression_datastore:
                futures.append(executor.submit(process_patient,
                                               gene_expression_patient,
                                               overall_survival_datastore,
                                               whole_slide_image_datastore,
                                               gene_expression_keys))

            # Process each future as it completes
            for future in as_completed(futures):
                result = future.result()
                if result:
                    dataset.append(result)
                pbar.update(1)  # Update the progress bar for each completed task
    print(f"Loaded {len(dataset)} samples")

    # Storing dataset inside a CSV file
    csv_storer(table_paths[GENE_EXPRESSION_KEYS], gene_expression_keys, ["gene"], keys_mode=True)
    gene_expression_keys = (['case_id', 'slide_id', 'patient', 'survival_months', 'censorship'] +
                            [key + '_rnaseq' for key in gene_expression_keys])
    dataframe = pd.DataFrame(dataset, columns=gene_expression_keys)
    dataframe.to_csv(dataset_paths[GENE_EXPRESSION_MCAT], index=True)
    print(f"Data has been correctly saved inside {dataset_paths[GENE_EXPRESSION_MCAT]} file")

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
