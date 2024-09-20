from collections import Counter
from config.methods.configuration_loader import *
from json_dir.methods.json_loader import *

### CONFIGURATION
JSON_PATH_YAML = '../../config/files/json_path.yaml'
GENE_ESPRESSION_JSON = 'gene_expression_json'
METHYLATION_JSON = 'methylation_json'
OVERALL_SURVIVAL_JSON = 'overall_survival_json'


## FUNCTIONS
def myprint(dictionary):
    i = 1
    for element in dictionary:
        print(i, element)
        i += 1

def check_duplicates_case_ids(dictionary, json):
    case_ids = []
    for element in data[json]:
        case_ids.append(element['case_id'])

    from collections import Counter
    duplicates = []
    for element, count in Counter(case_ids).items():
        if count > 1:
            duplicates.append([element, count])
    print("Duplicates case IDs:", duplicates)
    print(len(duplicates))

def check_singles_case_ids(dictionary, json):
    case_ids = []
    for element in data[json]:
        case_ids.append(element['case_id'])

    singles = []
    for element, count in Counter(case_ids).items():
        if count == 1:
            singles.append([element, count])
    print("Singles case IDs:", singles)
    print(len(singles))


## MAIN
if __name__ == "__main__":

    # Loading YAML file
    file_path = yaml_loader(JSON_PATH_YAML)

    # Storing data from JSON file
    data = {GENE_ESPRESSION_JSON: json_loader(file_path[GENE_ESPRESSION_JSON]),
            METHYLATION_JSON: json_loader(file_path[METHYLATION_JSON]),
            OVERALL_SURVIVAL_JSON: json_loader(file_path[OVERALL_SURVIVAL_JSON])}

    # Storing only information about 'case_id', 'file_name' and 'file_id' for each JSON file
    for file in data:
        buffer = []
        for item in data[file]:
            buffer.append(
                {'case_id': item['cases'][0]['case_id'], 'file_name': item['file_name'], 'file_id': item['file_id']})
        data[file] = buffer

    '''
    # FIRST CASE: Overall Survival based only on Gene Expression
    overall_survival_ids = {record['case_id'] for record in data['overall_survival_json']}
    # Filtra i record in gene_expression_json che hanno un case_id presente in overall_survival_json
    filtered_gene_expression = [record for record in data['gene_expression_json'] if record['case_id'] in overall_survival_ids]
    '''
