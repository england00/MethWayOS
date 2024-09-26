from config.methods.configuration_loader import *
from json_dir.methods.json_loader import *
from json_dir.methods.json_storer import *
from data.methods.directory_loader import *
from data.methods.tsv_loader import *


## CONFIGURATION
JSON_PATHS_YAML = '../../config/files/json_paths.yaml'
DIRECTORIES_PATHS_YAML = '../../config/files/directories_paths.yaml'
DATASTORE_PATHS_YAML = '../../config/files/datastore_paths.yaml'
GENE_EXPRESSION = 'gene_expression'
OVERALL_SURVIVAL = 'overall_survival'


## MAIN
if __name__ == "__main__":
    
    # Loading YAML files
    json_paths = yaml_loader(JSON_PATHS_YAML)
    directories_paths = yaml_loader(DIRECTORIES_PATHS_YAML)
    datastore_paths = yaml_loader(DATASTORE_PATHS_YAML)

    # Storing OVERALL SURVIVAL data from JSON file
    overall_survival_list = json_loader(datastore_paths[OVERALL_SURVIVAL])
    case_ids = []
    for dictionary in overall_survival_list:
        case_ids.append(dictionary['info']['case_id'])

    # Storing GENE EXPRESSION data from JSON file with 'case_id', 'file_name' and 'file_id' only for interested cases
    gene_expression_list = json_loader(json_paths[GENE_EXPRESSION])
    buffer = []
    for dictionary in gene_expression_list:
        if dictionary['cases'][0]['case_id'] in case_ids:
            buffer.append({'case_id': dictionary['cases'][0]['case_id'], 'file_name': dictionary['file_name'], 'file_id': dictionary['file_id']})
    gene_expression_list = buffer

    # Removing duplicates, taking only one case_id for each-one
    buffer = []
    case_ids = []
    file_names = []
    for dictionary in gene_expression_list:
        if dictionary["case_id"] not in case_ids:
            buffer.append(dictionary)
            case_ids.append(dictionary["case_id"])
            file_names.append(dictionary['file_name'])
    gene_expression_list = buffer

    # Searching only TSV files with the right 'case_id'
    i = 0
    gene_expression_datastore = []
    for path in directory_loader(directories_paths[GENE_EXPRESSION]):
        name = path.split('/')[len(path.split('/')) - 1]
        if name in file_names:
            for dictionary in gene_expression_list:
                if name == dictionary['file_name']:
                    i += 1
                    gene_expression_datastore.append(tsv_loader(path, dictionary))
                    break
    print(f"Loaded {i} files")

    # Storing the datastore inside a JSON file
    json_storer(datastore_paths[GENE_EXPRESSION], gene_expression_datastore)
