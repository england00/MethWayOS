from collections import Counter
from config.methods.configuration_loader import *
from json_dir.methods.json_loader import *

### CONFIGURATION
JSON_PATHS_YAML = '../../config/files/json_paths.yaml'
DIRECTORIES_PATHS_YAML = '../../config/files/directories_paths.yaml'
DATASET_PATHS_YAML = '../../config/files/dataset_paths.yaml'
GENE_EXPRESSION = 'gene_expression'
METHYLATION = 'methylation'
OVERALL_SURVIVAL = 'overall_survival'


## FUNCTIONS
def myprint(dictionary):
    i = 1
    for element in dictionary:
        print(i, element)
        i += 1


def check_duplicates_case_ids(dictionary, json_file):
    case_ids = []
    for element in data[json_file]:
        case_ids.append(element['case_id'])

    from collections import Counter
    duplicates = []
    for element, count in Counter(case_ids).items():
        if count > 1:
            duplicates.append([element, count])
    print("Duplicates case IDs:", duplicates)
    print(len(duplicates))


def check_singles_case_ids(dictionary, json_file):
    case_ids = []
    for element in data[json_file]:
        case_ids.append(element['case_id'])

    singles = []
    for element, count in Counter(case_ids).items():
        if count == 1:
            singles.append([element, count])
    print("Singles case IDs:", singles)
    print(len(singles))


## MAIN
if __name__ == "__main__":

    # Loading YAML files
    json_paths = yaml_loader(JSON_PATHS_YAML)
    directories_paths = yaml_loader(DIRECTORIES_PATHS_YAML)
    dataset_paths = yaml_loader(DATASET_PATHS_YAML)



    print(json_paths)

    '''

    # Storing data from JSON file
    data = {GENE_EXPRESSION: json_loader(json_paths[GENE_EXPRESSION]),
            METHYLATION: json_loader(json_paths[METHYLATION]),
            OVERALL_SURVIVAL: json_loader(json_paths[OVERALL_SURVIVAL])}

    # Storing only information about 'case_id', 'file_name' and 'file_id' for each JSON file
    for file in data:
        buffer = []
        for item in data[file]:
            buffer.append(
                {'case_id': item['cases'][0]['case_id'], 'file_name': item['file_name'], 'file_id': item['file_id']})
        data[file] = buffer

    myprint(data[OVERALL_SURVIVAL])
    '''
