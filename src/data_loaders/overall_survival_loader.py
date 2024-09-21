from config.methods.configuration_loader import *
from json_dir.methods.json_storer import *
from data.methods.directory_loader import *
from data.methods.xml_loader import *

### CONFIGURATION
JSON_PATHS_YAML = '../../config/files/json_paths.yaml'
DIRECTORIES_PATHS_YAML = '../../config/files/directories_paths.yaml'
DATASET_PATHS_YAML = '../../config/files/dataset_paths.yaml'
OVERALL_SURVIVAL = 'overall_survival'

## MAIN
if __name__ == "__main__":

    # Loading YAML files
    json_paths = yaml_loader(JSON_PATHS_YAML)
    directories_paths = yaml_loader(DIRECTORIES_PATHS_YAML)
    dataset_paths = yaml_loader(DATASET_PATHS_YAML)

    # Loading XML files
    file_paths = directory_loader(directories_paths[OVERALL_SURVIVAL])
    overall_survival_list = []
    for path in file_paths:
        if xml_loader(path) is not None:
            overall_survival_list.append(xml_loader(path))

    # Selecting only DEAD cases
    buffer = []
    for dictionary in overall_survival_list:
        if dictionary['last_check']['vital_status'] == 'Dead':
            buffer.append(dictionary)
    overall_survival_list = buffer
    json_storer(dataset_paths[OVERALL_SURVIVAL], overall_survival_list)
