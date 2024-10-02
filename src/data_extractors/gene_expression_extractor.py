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
GENE_EXPRESSION = 'gene_expression'
OVERALL_SURVIVAL = 'overall_survival'
LOG_PATH = '../../logs/files/2.1 - GENE EXPRESSION Extractor.txt'


## FUNCTIONS
def dictionary_format(file_path, data_dictionary):
    # Loading TSV file
    data = tsv_loader(file_path)

    # Formatting data inside a dictionary
    genes_list = data[6:]
    dict_buffer = {'info': data_dictionary}
    for element in genes_list:
        # Selecting only the coding genes with only 'tpm_unstranded', 'fpkm_unstranded' and	'fpkm_uq_unstranded'
        if element[2] == 'protein_coding':
            dict_buffer[str(element[0])] = [element[6], element[7], element[8]]

            '''
            # Selecting all the genes with all the information (buffer as a list of dictionaries)
            buffer.append({str(data[1][0]): element[0],  # gene_id
                           str(data[1][1]): element[1],  # gene_name
                           str(data[1][2]): element[2],  # gene_type
                           str(data[1][3]): element[3],  # unstranded
                           str(data[1][4]): element[4],  # stranded_first
                           str(data[1][5]): element[5],  # stranded_second
                           str(data[1][6]): element[6],  # tpm_unstranded
                           str(data[1][7]): element[7],  # fpkm_unstranded
                           str(data[1][8]): element[8]})  # fpkm_uq_unstranded
            '''

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

    # Storing GENE EXPRESSION data from JSON file with 'case_id', 'file_name' and 'file_id' only for interested cases
    gene_expression_list = json_loader(json_paths[GENE_EXPRESSION])
    buffer = []
    for dictionary in gene_expression_list:
        if dictionary['cases'][0]['case_id'] in case_ids:
            buffer.append({'case_id': dictionary['cases'][0]['case_id'],
                           'file_name': dictionary['file_name'],
                           'file_id': dictionary['file_id']})
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
                    gene_expression_datastore.append(dictionary_format(path, dictionary))
                    break
    print(f"Loaded {i} files")

    # Storing the datastore inside a JSON file
    json_storer(datastore_paths[GENE_EXPRESSION], gene_expression_datastore)

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
