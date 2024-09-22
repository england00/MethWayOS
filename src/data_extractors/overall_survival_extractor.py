from config.methods.configuration_loader import *
from json_dir.methods.json_loader import *
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

    # Storing OVERALL SURVIVAL data from JSON file (only 'case_id', 'file_name' and 'file_id')
    overall_survival_list = json_loader(json_paths[OVERALL_SURVIVAL])
    buffer = []
    for item in overall_survival_list:
        buffer.append(
            {'case_id': item['cases'][0]['case_id'], 'file_name': item['file_name'], 'file_id': item['file_id']})
    overall_survival_list = buffer

    # Loading all the XML files
    file_paths = directory_loader(directories_paths[OVERALL_SURVIVAL])
    overall_survival_dataset = []
    for path in file_paths:
        if xml_loader(path) is not None:
            overall_survival_dataset.append(xml_loader(path))

    # Selecting only DEAD cases
    buffer = []
    for dictionary in overall_survival_dataset:
        if dictionary['last_check']['vital_status'] == 'Dead':
            buffer.append(dictionary)
            # Adding for each dictionary 'case_id', 'file_name' and 'file_id'
            file_name = dictionary['info'].split('\\')[len(dictionary['info'].split('\\')) - 1]
            for item in overall_survival_list:
                if item['file_name'] == file_name:
                    dictionary['info'] = item
                    break
    overall_survival_dataset = buffer

    # Storing the dataset inside a JSON file
    json_storer(dataset_paths[OVERALL_SURVIVAL], overall_survival_dataset)
