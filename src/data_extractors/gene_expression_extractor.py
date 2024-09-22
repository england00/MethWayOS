from config.methods.configuration_loader import *
from json_dir.methods.json_loader import *
from json_dir.methods.json_storer import *
from data.methods.directory_loader import *

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


## MAIN
if __name__ == "__main__":

    # Loading YAML files
    json_paths = yaml_loader(JSON_PATHS_YAML)
    directories_paths = yaml_loader(DIRECTORIES_PATHS_YAML)
    dataset_paths = yaml_loader(DATASET_PATHS_YAML)

    # Storing OVERALL SURVIVAL data from JSON file
    overall_survival_list = json_loader(dataset_paths[OVERALL_SURVIVAL])
    case_ids = []
    for dictionary in overall_survival_list:
        case_ids.append(dictionary['info']['case_id'])

    # Storing GENE EXPRESSION data from JSON file (only 'case_id', 'file_name' and 'file_id')
    gene_expression_list = json_loader(json_paths[GENE_EXPRESSION])
    buffer = []
    for dictionary in gene_expression_list:
        if dictionary['cases'][0]['case_id'] in case_ids:
            buffer.append({'case_id': dictionary['cases'][0]['case_id'], 'file_name': dictionary['file_name'], 'file_id': dictionary['file_id']})
    gene_expression_list = buffer

    # Removing duplicates, taking only one case_id for each-one
    buffer = []
    case_ids = []
    for dictionary in gene_expression_list:
        case_id = dictionary["case_id"]
        if case_id not in case_ids:
            case_ids.append(case_id)
            buffer.append(dictionary)
    gene_expression_list = buffer

    myprint(gene_expression_list)

    file_paths = directory_loader(directories_paths[GENE_EXPRESSION])
    print(file_paths)

