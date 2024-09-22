from datetime import datetime
from config.methods.configuration_loader import *
from json_dir.methods.json_loader import *
from data.methods.csv_storer import *


### CONFIGURATION
DATASTORE_PATHS_YAML = '../../config/files/datastore_paths.yaml'
DATASET_PATH_YAML = '../../config/files/dataset_paths.yaml'
GENE_EXPRESSION = 'gene_expression'
OVERALL_SURVIVAL = 'overall_survival'


## MAIN
if __name__ == "__main__":

    # Loading YAML files
    datastore_paths = yaml_loader(DATASTORE_PATHS_YAML)
    dataset_paths = yaml_loader(DATASET_PATH_YAML)

    # Storing data from JSON datastore
    gene_expression_datastore = json_loader(datastore_paths[GENE_EXPRESSION])
    overall_survival_datastore = json_loader(datastore_paths[OVERALL_SURVIVAL])

    # Creating the dataset with GENE EXPRESSION as feature vector and OVERALL SURVIVAL as label
    gene_expression_keys = gene_expression_datastore[0].keys()
    i = 1
    dataset = []
    for patient in gene_expression_datastore:
        for case in overall_survival_datastore:
            if patient['info']['case_id'] == case['info']['case_id']:
                buffer = []
                for key in gene_expression_keys:
                    if key != 'info':
                        buffer.append(patient[key][2])
                buffer.append(case['last_check']['days_to_death'])
                dataset.append(buffer)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{timestamp} - Loaded row n°{i} of patient {case['info']['case_id']}")
                i += 1
                break

    # Storing dataset inside a CSV file
    csv_storer(dataset_paths[GENE_EXPRESSION], dataset)
