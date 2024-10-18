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
METHYLATION = 'methylation'
OVERALL_SURVIVAL = 'overall_survival'
METHYLATION_NAMES = 'methylation_names'
LOG_PATH = '../../logs/files/0 - METHYLATION Extractor.txt'


## FUNCTIONS
def dictionary_format(file_path, data_dictionary):
    # Loading TSV file
    data = tsv_loader(file_path)

    # Formatting data inside a dictionary
    dict_buffer = {'info': data_dictionary}
    for element in data:
        if str(element[0]).startswith("cg"):
            if element[1] != 'NA':
                dict_buffer[str(element[0])] = element[1]
            else:
                dict_buffer[str(element[0])] = 0.0

    return dict_buffer


## MAIN
if __name__ == "__main__":

    # Open LOG file
    logfile = open(LOG_PATH, 'w')
    sys.stdout = DualOutput(sys.stdout, logfile)

    # Loading YAML files
    json_paths = yaml_loader(JSON_PATHS_YAML)
    directories_paths = yaml_loader(DIRECTORIES_PATHS_YAML)
    datastore_paths = yaml_loader(DATASTORE_PATHS_YAML)

    # Storing OVERALL SURVIVAL data from JSON file
    overall_survival_list = json_loader(datastore_paths[OVERALL_SURVIVAL])
    case_ids = []
    for dictionary in overall_survival_list:
        case_ids.append(dictionary['info']['case_id'])

    # Storing METHYLATION data from JSON file with 'case_id', 'file_name' and 'file_id' only for interested cases
    methylation_list = json_loader(json_paths[METHYLATION])
    buffer = []
    for dictionary in methylation_list:
        if dictionary['cases'][0]['case_id'] in case_ids:
            buffer.append({'case_id': dictionary['cases'][0]['case_id'], 
                           'file_name': dictionary['file_name'],
                           'file_id': dictionary['file_id']})
    methylation_list = buffer

    # Removing duplicates, taking only one case_id for each-one
    buffer = []
    case_ids = []
    file_names = []
    for dictionary in methylation_list:
        if dictionary["case_id"] not in case_ids:
            buffer.append(dictionary)
            case_ids.append(dictionary["case_id"])
            file_names.append(dictionary['file_name'])
    methylation_list = buffer

    # Searching only TXT files with the right 'case_id'
    i = 0
    methylation_datastore = []
    for path in directory_loader(directories_paths[METHYLATION]):
        name = path.split('/')[len(path.split('/')) - 1]
        if name in file_names:
            for dictionary in methylation_list:
                if name == dictionary['file_name']:
                    i += 1
                    methylation_datastore.append(dictionary_format(path, dictionary))
                    break
    print(f"Loaded {i} files")

    # Storing the datastore inside a JSON file
    json_storer(datastore_paths[METHYLATION], methylation_datastore)

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
