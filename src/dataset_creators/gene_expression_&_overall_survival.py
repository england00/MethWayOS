from config.methods.configuration_loader import *
from json_dir.methods.json_loader import *
from json_dir.methods.json_storer import *
from data.methods.csv_storer import *
from logs.methods.log_storer import *


## CONFIGURATION
JSON_PATHS_YAML = '../../config/files/json_paths.yaml'
DATASTORE_PATHS_YAML = '../../config/files/datastore_paths.yaml'
DATASET_PATH_YAML = '../../config/files/dataset_paths.yaml'
GENE_EXPRESSION = 'gene_expression'
OVERALL_SURVIVAL = 'overall_survival'
GENE_EXPRESSION_NAMES = 'gene_expression_names'
LOG_PATH = '../../logs/files/2.2 - GENE EXPRESSION & OS - Dataset.txt'


## MAIN
if __name__ == "__main__":

    # Open LOG file
    logfile = open(LOG_PATH, 'w')
    sys.stdout = DualOutput(sys.stdout, logfile)

    # Loading YAML files
    json_paths = yaml_loader(JSON_PATHS_YAML)
    datastore_paths = yaml_loader(DATASTORE_PATHS_YAML)
    dataset_paths = yaml_loader(DATASET_PATH_YAML)

    # Storing data from JSON datastore
    gene_expression_datastore = json_loader(datastore_paths[GENE_EXPRESSION])
    overall_survival_datastore = json_loader(datastore_paths[OVERALL_SURVIVAL])

    # Creating the dataset with GENE EXPRESSION as feature vector and OVERALL SURVIVAL as label
    gene_expression_keys = [key for key in gene_expression_datastore[0].keys()]
    gene_expression_keys = gene_expression_keys[1:]
    i = 0
    dataset = []
    for patient in gene_expression_datastore:
        for case in overall_survival_datastore:
            if patient['info']['case_id'] == case['info']['case_id']:
                buffer = []
                for key in gene_expression_keys:
                    buffer.append(patient[key][2])  # Adding each feature
                if case['last_check']['vital_status'] == 'Dead':  # DEAD cases
                    buffer.append(case['last_check']['days_to_death'])  # Adding label
                else:  # ALIVE cases
                    buffer.append(case['last_check']['days_to_last_followup'])  # Adding label
                dataset.append(buffer)
                i += 1
                break
    print(f"Loaded {i} files")

    # Storing dataset inside a CSV file
    csv_storer(dataset_paths[GENE_EXPRESSION], dataset)
    json_storer(json_paths[GENE_EXPRESSION_NAMES], gene_expression_keys)

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
