import pandas as pd
from json_dir.methods.json_storer import *
from data.methods.directory_loader import *
from data.methods.csv_dataset_loader import *
from data.methods.tsv_loader import *
from logs.methods.log_storer import *


## CONFIGURATION
JSON_PATHS_YAML = '../../config/files/json_paths.yaml'
DIRECTORIES_PATHS_YAML = '../../config/files/directories_paths.yaml'
DATASTORE_PATHS_YAML = '../../config/files/datastore_paths.yaml'
TABLE_PATHS_YAML = '../../config/files/table_paths.yaml'
GENE_EXPRESSION = 'gene_expression'
GENE_EXPRESSION_TSS = 'gene_expression_tss'
GENE_EXPRESSION_TSS_NAMES = 'gene_expression_tss_names'
OVERALL_SURVIVAL = 'overall_survival'
LOG_PATH = '../../logs/files/0 - GENE EXPRESSION Extractor.txt'


## FUNCTIONS
def dictionary_format(file_path, data_dictionary, tss_dictionary):
    # Loading TSV file
    data = tsv_loader(file_path)

    # Formatting data inside a dictionary
    genes_list = data[6:]
    dict_buffer = {'info': data_dictionary}
    for element in genes_list:

        # Selecting only the coding genes with only 'tpm_unstranded', 'fpkm_unstranded' and	'fpkm_uq_unstranded'
        if element[2] == 'protein_coding':
            if str(element[0]).split('.')[0] in tss_dictionary:                                 # gene_id
                dict_buffer[str(element[0])] = [element[1],                                     # gene_name
                                                element[2],                                     # gene_type
                                                element[3],                                     # unstranded
                                                element[4],                                     # stranded_first
                                                element[5],                                     # stranded_second
                                                element[6],                                     # tpm_unstranded
                                                element[7],                                     # fpkm_unstranded
                                                element[8],                                     # fpkm_uq_unstranded
                                                str(tss_dictionary[str(element[0]).split('.')[0]])]  # TSS
            else:
                dict_buffer[str(element[0])] = [element[1],                                     # gene_name
                                                element[2],                                     # gene_type
                                                element[3],                                     # unstranded
                                                element[4],                                     # stranded_first
                                                element[5],                                     # stranded_second
                                                element[6],                                     # tpm_unstranded
                                                element[7],                                     # fpkm_unstranded
                                                element[8],                                     # fpkm_uq_unstranded
                                                None]                                           # TSS

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
    table_paths = yaml_loader(TABLE_PATHS_YAML)

    # Loading OVERALL SURVIVAL data from JSON file
    overall_survival_list = json_loader(datastore_paths[OVERALL_SURVIVAL])
    case_ids = []
    for dictionary in overall_survival_list:
        case_ids.append(dictionary['info']['case_id'])

    # Loading GENE EXPRESSION TSS data from CSV file
    gene_expression_tss_dataframe = pd.read_csv(table_paths[GENE_EXPRESSION_TSS])
    gene_expression_tss_dictionary = gene_expression_tss_dataframe.set_index('gene_id')['TSS'].to_dict()

    # Loading GENE EXPRESSION data from JSON file with 'case_id', 'file_name' and 'file_id' only for interested cases
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
                    gene_expression_datastore.append(
                        dictionary_format(path, dictionary, gene_expression_tss_dictionary))
                    break
    print(f"Loaded {i} files")

    # Storing the datastore inside a JSON file
    json_storer(datastore_paths[GENE_EXPRESSION], gene_expression_datastore)

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
