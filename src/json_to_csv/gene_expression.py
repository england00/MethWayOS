import pandas as pd
from config.methods.configuration_loader import *
from json_dir.methods.json_loader import *


## CONFIGURATION
JSON_PATHS_YAML = '../../config/files/json_paths.yaml'
DATASTORE_PATHS_YAML = '../../config/files/datastore_paths.yaml'
TABLE_PATHS_YAML = '../../config/files/table_paths.yaml'
GENE_EXPRESSION = 'gene_expression'


## MAIN
if __name__ == "__main__":
    # Loading YAML files
    json_paths = yaml_loader(JSON_PATHS_YAML)
    datastore_paths = yaml_loader(DATASTORE_PATHS_YAML)
    table_paths = yaml_loader(TABLE_PATHS_YAML)

    # Loading data from JSON datastore and giving DataFrame format
    gene_expression_datastore = json_loader(datastore_paths[GENE_EXPRESSION])
    gene_expression_dataframe = pd.DataFrame(gene_expression_datastore)

    # Storing dataset inside a CSV file
    gene_expression_dataframe.to_csv(table_paths[GENE_EXPRESSION], index=False)
