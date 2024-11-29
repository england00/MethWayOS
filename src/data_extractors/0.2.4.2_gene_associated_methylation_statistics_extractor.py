import ast
import copy
import numpy as np
import pandas as pd
from config.methods.configuration_loader import *
from json_dir.methods.json_loader import *
from json_dir.methods.json_storer import *
from data.methods.directory_loader import *
from data.methods.tsv_loader import *
from logs.methods.log_storer import *


## CONFIGURATION
JSON_PATHS_YAML = '../../config/files/json_paths.yaml'
DIRECTORIES_PATHS_YAML = '../../config/files/directories_paths.yaml'
DATASTORE_PATHS_YAML = '../../config/files/datastore_paths.yaml'
TABLE_PATHS_YAML = '../../config/files/table_paths.yaml'
METHYLATION = 'methylation'
GENE_ASSOCIATED_METHYLATION_STATISTICS = 'gene_associated_methylation_statistics'
METHYLATION_VECTORS_FOR_EACH_GENE_FULL = 'methylation_vectors_for_each_gene_full'
SELECTED_METHYLATION_ISLANDS_FULL = 'selected_methylation_islands_full'
OVERALL_SURVIVAL = 'overall_survival'
METHYLATION_NAMES = 'methylation_names'
LOG_PATH = f'../../logs/files/{os.path.basename(__file__)}.txt'


## FUNCTIONS
def is_numeric(value):
    # Checking numeric values
    try:
        float(value)
        return True
    except ValueError:
        return False


def compute_statistics(values):
    # Computing statistics
    numeric_values = [float(x) for x in values if is_numeric(x)]
    if numeric_values:
        max_value = np.max(numeric_values)
        mean_value = np.mean(numeric_values)
        std_dev = np.std(numeric_values)
        total_sum = np.sum(numeric_values)
        non_zero_percentage = (np.count_nonzero(numeric_values) / len(numeric_values)) * 100
        return [max_value, mean_value, std_dev, total_sum, non_zero_percentage]
    return None


def dictionary_format(file_path, patient_dictionary, selected_islands_dictionary, islands_vectors_dictionary):
    # Loading TSV file
    data = tsv_loader(file_path)
    buffer_dictionary = copy.deepcopy(islands_vectors_dictionary)

    # Formatting data inside a dictionary
    datastore_item = {'info': patient_dictionary}
    for element in data:
        if str(element[0]) in selected_islands_dictionary:
            for gene in selected_islands_dictionary[str(element[0])]:
                for index in range(len(islands_vectors_dictionary[gene])):
                    if islands_vectors_dictionary[gene][index] == str(element[0]):
                        if element[1] != 'NA':
                            buffer_dictionary[gene][index] = element[1]
                        else:
                            buffer_dictionary[gene][index] = 0.0

    # Keeping only numerical values and computing statistics
    buffer_dictionary = {key: compute_statistics(values) for key, values in buffer_dictionary.items()}
    buffer_dictionary = {key: values for key, values in buffer_dictionary.items() if values is not None}

    # Storing values for datastore
    for key in buffer_dictionary:
        datastore_item[str(key + "_max")] = buffer_dictionary[key][0]
        datastore_item[str(key + "_mean")] = buffer_dictionary[key][1]
        datastore_item[str(key + "_std")] = buffer_dictionary[key][2]
        datastore_item[str(key + "_sum")] = buffer_dictionary[key][3]
        datastore_item[str(key + "_non_zero_percentage")] = buffer_dictionary[key][4]
    del buffer_dictionary

    return datastore_item


## MAIN
if __name__ == "__main__":

    # Open LOG file
    logfile = open(LOG_PATH, 'w')
    sys.stdout = DualOutput(sys.stdout, logfile)

    # Loading YAML files
    json_paths = yaml_loader(JSON_PATHS_YAML)
    directories_paths = yaml_loader(DIRECTORIES_PATHS_YAML)
    datastore_paths = yaml_loader(DATASTORE_PATHS_YAML)
    table_paths = yaml_loader(TABLE_PATHS_YAML)

    # Loading OVERALL SURVIVAL data from JSON file
    overall_survival_list = json_loader(datastore_paths[OVERALL_SURVIVAL])
    case_ids = []
    for patient in overall_survival_list:
        case_ids.append(patient['info']['case_id'])

    # Loading METHYLATION data from JSON file with 'case_id', 'file_name' and 'file_id' only for interested cases
    methylation_list = json_loader(json_paths[METHYLATION])
    buffer = []
    for patient in methylation_list:
        if patient['cases'][0]['case_id'] in case_ids:
            buffer.append({'case_id': patient['cases'][0]['case_id'],
                           'file_name': patient['file_name'],
                           'file_id': patient['file_id']})
    methylation_list = buffer

    # Removing duplicates, taking only one case_id for each-one
    buffer = []
    case_ids = []
    file_names = []
    for patient in methylation_list:
        if patient["case_id"] not in case_ids:
            buffer.append(patient)
            case_ids.append(patient["case_id"])
            file_names.append(patient['file_name'])
    methylation_list = buffer

    # Loading METHYLATION Vectors and storing into a dictionary
    methylation_vectors_list = pd.read_csv(table_paths[METHYLATION_VECTORS_FOR_EACH_GENE_FULL])
    print(f"Data has been correctly loaded from {table_paths[METHYLATION_VECTORS_FOR_EACH_GENE_FULL]} file")
    methylation_vectors_list['beta_values'] = methylation_vectors_list['beta_values'].apply(ast.literal_eval)
    methylation_vectors_dictionary = methylation_vectors_list.set_index('gene_name')['beta_values'].to_dict()
    methylation_vectors_dictionary_filtered = {key: [x for x in vector if x != 0] for key, vector in methylation_vectors_dictionary.items() if any(x != 0 for x in vector)}

    # Searching only TXT files with the right 'case_id', also selecting some chosen islands to store
    dictionary_selected_methylation_islands = json_loader(datastore_paths[SELECTED_METHYLATION_ISLANDS_FULL])
    i = 0
    methylation_datastore = []
    for path in directory_loader(directories_paths[METHYLATION]):
        name = path.split('/')[len(path.split('/')) - 1]
        if name in file_names:
            for patient in methylation_list:
                if name == patient['file_name']:
                    i += 1
                    methylation_datastore.append(dictionary_format(path,
                                                                   patient,
                                                                   dictionary_selected_methylation_islands,
                                                                   methylation_vectors_dictionary_filtered))
                    break
    print(f"Loaded {i} files")

    # Storing the datastore inside a JSON file
    json_storer(datastore_paths[GENE_ASSOCIATED_METHYLATION_STATISTICS], methylation_datastore)

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
