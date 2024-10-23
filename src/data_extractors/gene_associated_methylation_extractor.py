from config.methods.configuration_loader import *
from json_dir.methods.json_loader import *
from json_dir.methods.json_storer import *
from data.methods.tsv_loader import *
from logs.methods.log_storer import *
import concurrent.futures
from tqdm import tqdm  # Importa tqdm per la barra di progresso

## CONFIGURATION
JSON_PATHS_YAML = '../../config/files/json_paths.yaml'
DIRECTORIES_PATHS_YAML = '../../config/files/directories_paths.yaml'
DATASTORE_PATHS_YAML = '../../config/files/datastore_paths.yaml'
CPG950_CODING_GENES = 'cpg950_coding_genes'
GENE_ASSOCIATED_METHYLATION = 'gene_associated_methylation'
METHYLATION = 'methylation'
OVERALL_SURVIVAL = 'overall_survival'
LOG_PATH = '../../logs/files/0 - GENE ASSOCIATED METHYLATION Extractor.txt'


## FUNCTIONS
def dictionary_format(file_path, data_dictionary):
    # Loading TSV file
    data = tsv_loader(file_path)

    # Formatting data inside a dictionary
    dict_buffer = {'info': data_dictionary}
    for element in data:
        if str(element[0]).startswith("cg"):
            if element[1] != 'NA':
                dict_buffer[str(element[0])] = element[1]
            else:
                dict_buffer[str(element[0])] = 0.0

    return dict_buffer


def filtered_dictionary(dictionary, allowed_keys):
    return {key: dictionary[key] for key in allowed_keys if key in dictionary}


def process_dictionary(dictionary, allowed_keys):
    return filtered_dictionary(dictionary, allowed_keys)


## MAIN
if __name__ == "__main__":

    # Open LOG file
    logfile = open(LOG_PATH, 'w')
    sys.stdout = DualOutput(sys.stdout, logfile)

    # Loading YAML files
    json_paths = yaml_loader(JSON_PATHS_YAML)
    directories_paths = yaml_loader(DIRECTORIES_PATHS_YAML)
    datastore_paths = yaml_loader(DATASTORE_PATHS_YAML)

    # Loading data from JSON datastores
    methylation_datastore = json_loader(datastore_paths[METHYLATION])
    cpg950_coding_genes_datastore = json_loader(datastore_paths[CPG950_CODING_GENES])

    # Loading CPG950 CODING GENES data from JSON file
    cpg950_coding_genes_keys = [key for key in cpg950_coding_genes_datastore.keys()]
    cpg950_coding_genes_keys.append('info')

    # Using TQDM for Progress Bar
    allowed_keys = set(cpg950_coding_genes_keys)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        filtered_data = list(
            tqdm(executor.map(process_dictionary, methylation_datastore, [allowed_keys] * len(methylation_datastore)),
                 total=len(methylation_datastore),
                 desc="Filtering dictionaries"))

    # Storing the datastore inside a JSON file
    json_storer(datastore_paths[GENE_ASSOCIATED_METHYLATION], filtered_data)

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
