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
LOG_PATH = '../../logs/files/3.1 - METHYLATION Extractor.txt'


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

    print(len(case_ids))

    # Storing GENE EXPRESSION data from JSON file with 'case_id', 'file_name' and 'file_id' only for interested cases
    methylation_list = json_loader(json_paths[METHYLATION])
    buffer = []
    for dictionary in methylation_list:
        if dictionary['cases'][0]['case_id'] in case_ids:
            buffer.append({'case_id': dictionary['cases'][0]['case_id'], 
                           'file_name': dictionary['file_name'],
                           'file_id': dictionary['file_id']})
    methylation_list = buffer

    print(len(methylation_list))

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
