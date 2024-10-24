from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from config.methods.configuration_loader import *
from json_dir.methods.json_loader import *
from json_dir.methods.json_storer import *
from data.methods.csv_dataset_storer import *
from logs.methods.log_storer import *


## CONFIGURATION
JSON_PATHS_YAML = '../../config/files/json_paths.yaml'
DATASTORE_PATHS_YAML = '../../config/files/datastore_paths.yaml'
DATASET_PATH_YAML = '../../config/files/dataset_paths.yaml'
GENE_EXPRESSION = 'gene_expression'
METHYLATION = 'methylation'
OVERALL_SURVIVAL = 'overall_survival'
GENE_EXPRESSION_AND_METHYLATION = 'gene_expression_and_methylation'
GENE_EXPRESSION_AND_METHYLATION_NAMES = 'gene_expression_and_methylation_names'
LOG_PATH = '../../logs/files/1 - GENE EXPRESSION & METHYLATION & OS - Dataset.txt'


## FUNCTIONS
def process_patient(ge_patient, meth_datastore, os_datastore, ge_keys, meth_keys):
    for methylation_patient in meth_datastore:
        for patient in os_datastore:
            if (ge_patient['info']['case_id'] == methylation_patient['info']['case_id'] and
                    ge_patient['info']['case_id'] == patient['info']['case_id']):
                buffer = []
                # Gene Expression
                for key in ge_keys:
                    buffer.append(ge_patient[key][7])  # Adding each feature
                # Methylation
                for key in meth_keys:
                    buffer.append(methylation_patient[key])  # Adding each feature
                # Label
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

    # Loading YAML files
    json_paths = yaml_loader(JSON_PATHS_YAML)
    datastore_paths = yaml_loader(DATASTORE_PATHS_YAML)
    dataset_paths = yaml_loader(DATASET_PATH_YAML)

    # Storing data from JSON datastores
    gene_expression_datastore = json_loader(datastore_paths[GENE_EXPRESSION])
    methylation_datastore = json_loader(datastore_paths[METHYLATION])
    overall_survival_datastore = json_loader(datastore_paths[OVERALL_SURVIVAL])

    # Creating the dataset with GENE EXPRESSION and METHYLATION as feature vector and OVERALL SURVIVAL as label
    gene_expression_keys = [key for key in gene_expression_datastore[0].keys()][1:]
    methylation_keys = [key for key in methylation_datastore[0].keys()][1:]
    num_patients = len(gene_expression_datastore)
    dataset = []

    # Using ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        futures = []
        with tqdm(total=num_patients, desc="Processing cases") as pbar:  # Progress bar
            for gene_expression_patient in gene_expression_datastore:
                futures.append(executor.submit(process_patient,
                                               gene_expression_patient,
                                               methylation_datastore,
                                               overall_survival_datastore,
                                               gene_expression_keys,
                                               methylation_keys))

            # Process each future as it completes
            for future in as_completed(futures):
                result = future.result()
                if result:
                    dataset.append(result)
                pbar.update(1)  # Update the progress bar for each completed task
    print(f"Loaded {len(dataset)} samples")

    # Storing dataset inside a CSV file
    gene_expression_and_methylation_keys = gene_expression_keys + methylation_keys
    csv_storer(dataset_paths[GENE_EXPRESSION_AND_METHYLATION], dataset)
    json_storer(json_paths[GENE_EXPRESSION_AND_METHYLATION_NAMES], gene_expression_and_methylation_keys)

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
