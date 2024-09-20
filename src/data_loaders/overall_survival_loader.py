from collections import Counter
from config.methods.configuration_loader import *
from json_dir.methods.json_loader import *

### CONFIGURATION
JSON_PATH_YAML = '../../config/files/json_paths.yaml'
OVERALL_SURVIVAL = 'overall_survival'


## FUNCTIONS
def myprint(dictionary):
    i = 1
    for element in dictionary:
        print(i, element)
        i += 1


## MAIN
if __name__ == "__main__":

    # Loading YAML files
    json_path = yaml_loader(JSON_PATH_YAML)
    #directories_path = yaml_loader(OVERALL_SURVIVAL)

    # Storing data from JSON file
    data = json_loader(json_path[OVERALL_SURVIVAL])

    # Storing only information about 'case_id', 'file_name' and 'file_id' for the JSON file
    buffer = []
    for item in data:
        buffer.append(
            {'case_id': item['cases'][0]['case_id'], 'file_name': item['file_name'], 'file_id': item['file_id']})
    data = buffer

    # myprint(data)

    import os
