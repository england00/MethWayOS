import os
from collections import Counter
from config.methods.configuration_loader import *
from json_dir.methods.json_loader import *
from logs.methods.log_storer import *


## CONFIGURATION
''' General '''
JSON_PATHS_YAML = '../../config/paths/json_paths.yaml'
LOG_PATH = f'../../logs/files/{os.path.basename(__file__)}.txt'
CANCER_TYPE = 'GBMLGG'   # [BRCA, GBMLGG]

''' Input JSON file'''
GENE_EXPRESSION = 'gene_expression'
METHYLATION = 'methylation'
OVERALL_SURVIVAL = 'overall_survival'


## FUNCTIONS
def myprint(dictionary):
    i = 1
    for element in dictionary:
        print(i, element)
        i += 1


def check_duplicates_case_ids(json_file):
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


def check_singles_case_ids(json_file):
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

    # Open LOG file
    logfile = open(LOG_PATH, 'w')
    sys.stdout = DualOutput(sys.stdout, logfile)

    # Loading YAML file
    json_paths = yaml_loader(JSON_PATHS_YAML)

    # Loading data from JSON file
    data = {GENE_EXPRESSION: json_loader(json_paths[GENE_EXPRESSION]),
            METHYLATION: json_loader(json_paths[METHYLATION]),
            OVERALL_SURVIVAL: json_loader(json_paths[OVERALL_SURVIVAL])}

    # Storing only information about 'case_id', 'file_name' and 'file_id' for each JSON file
    for file in data:
        buffer = []
        for item in data[file]:
            if CANCER_TYPE == 'BRCA':
                buffer.append({'case_id': item['cases'][0]['case_id'], 'file_name': item['file_name'],
                               'file_id': item['file_id']})
            elif CANCER_TYPE == 'GBMLGG':
                buffer.append({'case_id': item['associated_entities'][0]['case_id'], 'file_name': item['file_name'], 'file_id': item['file_id']})
        data[file] = buffer

    # Printing data
    myprint(data[OVERALL_SURVIVAL])

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
