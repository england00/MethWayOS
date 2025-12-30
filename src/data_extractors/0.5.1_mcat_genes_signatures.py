import pandas as pd
from config.methods.configuration_loader import *
from data.methods.directory_loader import *
from logs.methods.log_storer import *


## CONFIGURATION
''' General '''
JSON_PATHS_YAML = '../../config/paths/json_paths.yaml'
LOG_PATH = f'../../logs/files/{os.path.basename(__file__)}.txt'
TABLE_PATHS_YAML = '../../config/paths/table_paths.yaml'

''' Input Keys'''
GENE_EXPRESSION_KEYS_METHYLATION = 'gene_expression_keys_methylation'
MCAT_SIGNATURES = 'mcat_signatures'
METHWAYOS_SIGNATURES = 'methwayos_signatures'

''' Output Keys'''
GENE_EXPRESSION_KEYS_METHYLATION_TRANSFORMED = 'gene_expression_keys_methylation_transformed'
MCAT_SIGNATURES_TRANSFORMED = 'mcat_signatures_transformed'
METHWAYOS_SIGNATURES_TRANSFORMED = 'methwayos_signatures_transformed'


## MAIN
def transform_gene_csv(input_file, output_file):
    dataframe = pd.read_csv(input_file)
    genes = dataframe.values.flatten()
    unique_genes = list(dict.fromkeys([gene for gene in genes if pd.notna(gene)]))
    dataframe_transformed = pd.DataFrame([unique_genes], columns=unique_genes)
    dataframe_transformed.to_csv(output_file, index=False)
    print(f"Data has been correctly saved inside {output_file} file")

## MAIN
if __name__ == "__main__":

    # Open LOG file
    logfile = open(LOG_PATH, 'w')
    sys.stdout = DualOutput(sys.stdout, logfile)

    # Loading YAML paths
    json_paths = yaml_loader(JSON_PATHS_YAML)
    table_paths = yaml_loader(TABLE_PATHS_YAML)

    # Executing transformation and storing
    transform_gene_csv(table_paths[GENE_EXPRESSION_KEYS_METHYLATION], table_paths[GENE_EXPRESSION_KEYS_METHYLATION_TRANSFORMED])

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
