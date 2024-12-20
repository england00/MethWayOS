from config.methods.configuration_loader import *
from json_dir.methods.json_loader import *
from json_dir.methods.json_storer import *
from data.methods.directory_loader import *
from data.methods.tsv_loader import *
from logs.methods.log_storer import *


## CONFIGURATION
DATASTORE_PATHS_YAML = '../../config/paths/datastore_paths.yaml'
DIRECTORIES_PATHS_YAML = '../../config/paths/directories_paths.yaml'
GENE_ASSOCIATED_METHYLATION = 'gene_associated_methylation'
JSON_PATHS_YAML = '../../config/paths/json_paths.yaml'
LOG_PATH = f'../../logs/files/{os.path.basename(__file__)}.txt'
METHYLATION = 'methylation'
METHYLATION_NAMES = 'methylation_names'
OVERALL_SURVIVAL = 'overall_survival'
SELECTED_METHYLATION_ISLANDS_FULL = 'selected_methylation_islands_full'


## FUNCTIONS
def dictionary_format(file_path, patient_dictionary, selected_islands_dictionary):
    # Loading TSV file
    data = tsv_loader(file_path)

    # Formatting data inside a dictionary
    dict_buffer = {'info': patient_dictionary}
    for element in data:
        if str(element[0]) in selected_islands_dictionary:
            if element[1] != 'NA':
                dict_buffer[str(element[0])] = element[1]
            else:
                dict_buffer[str(element[0])] = 0.0

    return dict_buffer


def common_keys_filter(data):
    # Finding common keys among all dictionaries
    common_keys = set.intersection(*(set(dictionary.keys()) for dictionary in data))

    return [{key: dictionary[key] for key in common_keys} for dictionary in data]


## MAIN
if __name__ == "__main__":

    # Open LOG file
    logfile = open(LOG_PATH, 'w')
    sys.stdout = DualOutput(sys.stdout, logfile)

    # Loading YAML paths
    json_paths = yaml_loader(JSON_PATHS_YAML)
    directories_paths = yaml_loader(DIRECTORIES_PATHS_YAML)
    datastore_paths = yaml_loader(DATASTORE_PATHS_YAML)

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

    # Searching only TXT paths with the right 'case_id', also selecting some chosen islands to store
    dictionary_selected_methylation_islands = json_loader(datastore_paths[SELECTED_METHYLATION_ISLANDS_FULL])
    i = 0
    methylation_datastore = []
    for path in directory_loader(directories_paths[METHYLATION]):
        name = path.split('/')[len(path.split('/')) - 1]
        if name in file_names:
            for patient in methylation_list:
                if name == patient['file_name']:
                    i += 1
                    methylation_datastore.append(dictionary_format(path, patient, dictionary_selected_methylation_islands))
                    break
    print(f"Loaded {i} paths")

    # Filtering only common keys
    methylation_datastore = common_keys_filter(methylation_datastore)

    # Storing the datastore inside a JSON file
    json_storer(datastore_paths[GENE_ASSOCIATED_METHYLATION], methylation_datastore)

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
