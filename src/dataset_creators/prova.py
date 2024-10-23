import pandas as pd
from tqdm import tqdm
from config.methods.configuration_loader import *
from json_dir.methods.json_loader import *
from json_dir.methods.json_storer import *
from data.methods.csv_storer import *
from logs.methods.log_storer import *

import json


## CONFIGURATION
JSON_PATHS_YAML = '../../config/files/json_paths.yaml'
DATASTORE_PATHS_YAML = '../../config/files/datastore_paths.yaml'
DATASET_PATH_YAML = '../../config/files/dataset_paths.yaml'
GENE_EXPRESSION = 'gene_expression'
METHYLATION = 'methylation'
OVERALL_SURVIVAL = 'overall_survival'
CPG950_CODING_GENES = 'cpg950_coding_genes'
GENE_EXPRESSION_WITH_ASSOCIATED_METHYLATION = 'gene_expression_with_associated_methylation'
GENE_EXPRESSION_WITH_ASSOCIATED_METHYLATION_NAMES = 'gene_expression_with_associated_methylation_names'
LOG_PATH = '../../logs/files/1 - GENE EXPRESSION & ASSOCIATED METHYLATION & OS - Dataset.txt'


def extract_methylation_vectors(df_meth, df_rna_result, data_cpg, size_vector, symmetrical_mode, filtro_meth='median',
                                filtro_gene='log_median'):
    TSS_index = int((131328 / 2) - 1)

    df_meth_val_dict = df_meth.set_index('ID_site')[filtro_meth].to_dict()

    # Create an empty DataFrame
    data = []

    if not symmetrical_mode:
        end_index_vect = TSS_index
        start_index_vect = int(end_index_vect - size_vector + 1)
    elif symmetrical_mode:
        center_index = TSS_index
        start_index_vect = int(center_index - (size_vector / 2) + 1)
        end_index_vect = int(center_index + (size_vector / 2))

    for i, gene_info in tqdm(df_rna_result.iterrows(), total=len(df_rna_result), desc="Estrazione vettori Beta_values"):
        gene_expression = gene_info[filtro_gene]
        vector = [0] * size_vector  # Create a list of zeros
        gene_name = gene_info['gene_name']
        gene_id = gene_info['gene_id']

        if gene_name in data_cpg:
            for island in data_cpg[gene_name]:
                index = int(island['site_index'])
                site_id = island['site_id']

                if start_index_vect <= index <= end_index_vect and site_id in df_meth_val_dict:
                    vector[index - start_index_vect] = df_meth_val_dict[site_id]

        # Append the row to the data list
        data.append([gene_name, gene_id, gene_expression, vector])

    # Convert the data list to a DataFrame
    data_df = pd.DataFrame(data, columns=['gene_name', 'gene_id', 'gene_expression', 'beta_values'])

    return data_df


## MAIN
if __name__ == "__main__":

    # Open LOG file
    logfile = open(LOG_PATH, 'w')
    sys.stdout = DualOutput(sys.stdout, logfile)

    # Loading YAML files
    json_paths = yaml_loader(JSON_PATHS_YAML)
    datastore_paths = yaml_loader(DATASTORE_PATHS_YAML)
    dataset_paths = yaml_loader(DATASET_PATH_YAML)

    # Storing data from JSON datastores
    gene_expression_datastore = json_loader(datastore_paths[GENE_EXPRESSION])
    methylation_datastore = json_loader(datastore_paths[METHYLATION])
    overall_survival_datastore = json_loader(datastore_paths[OVERALL_SURVIVAL])
    cpg950_coding_genes = json_loader(datastore_paths[CPG950_CODING_GENES])

    gene_expression_path = 'C:/Users/lucai/Downloads/df_gx_lungs.csv'
    methylation_path = 'C:/Users/lucai/Downloads/df_meth_lungs.csv'

    df_fpkm_uq = pd.read_csv(gene_expression_path)
    df_meth = pd.read_csv(methylation_path)[['ID_site', 'mean', 'median']]
    df_meth = df_meth[df_meth['ID_site'].str.startswith('cg')]

    data_meth = extract_methylation_vectors(df_meth, df_fpkm_uq, cpg950_coding_genes, size_vector=1000, symmetrical_mode=True,
                                            filtro_meth='median', filtro_gene='median')

    df_genes = df_fpkm_uq.merge(data_meth[['gene_id', 'beta_values']], on='gene_id', suffixes=('_df1', '_df2'),
                                how='inner')

    df_genes.to_csv('C:/Users/lucai/Downloads/file.csv', index=False)

    print(data_meth)
