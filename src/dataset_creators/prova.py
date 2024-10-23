from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from config.methods.configuration_loader import *
from json_dir.methods.json_loader import *
from json_dir.methods.json_storer import *
from data.methods.csv_storer import *
from logs.methods.log_storer import *

## CONFIGURATION
JSON_PATHS_YAML = '../../config/files/json_paths.yaml'
DATASTORE_PATHS_YAML = '../../config/files/datastore_paths.yaml'
DATASET_PATH_YAML = '../../config/files/dataset_paths.yaml'
CPG950_CODING_GENES = 'cpg950_coding_genes'
GENE_EXPRESSION = 'gene_expression'
GENE_EXPRESSION_WITH_ASSOCIATED_METHYLATION = 'gene_expression_with_associated_methylation'
METHYLATION = 'methylation'
OVERALL_SURVIVAL = 'overall_survival'
GENE_EXPRESSION_WITH_ASSOCIATED_METHYLATION_NAMES = 'gene_expression_with_associated_methylation_names'
LOG_PATH = '../../logs/files/1 - GENE EXPRESSION & ASSOCIATED METHYLATION & OS - Dataset.txt'


def process_patient(ge_patient, methylation_datastore, overall_survival_datastore, cpg950_coding_genes,
                    gene_expression_keys, methylation_keys):
    dataset = []
    for meth_patient in methylation_datastore:
        for case in overall_survival_datastore:
            # Matching Patient for GENE EXPRESSION, METHYLATION and OVERALL SURVIVAL
            if (ge_patient['info']['case_id'] == meth_patient['info']['case_id'] and
                    ge_patient['info']['case_id'] == case['info']['case_id']):
                buffer = []
                for key in gene_expression_keys:  # Gene Expression
                    # Checking if TSS value is not NONE and if the considered GENE is in the CPG950 datastore
                    if ge_patient[key][0] in cpg950_coding_genes and ge_patient[key][8] is not None:
                        buffer.append(ge_patient[key][7])  # Adding Gene Expression Value

                        tss_index = ge_patient[key][8]
                        window_size = 100  # Average Methylation Islands per Gene is 95
                        methylation_vector = []

                        counter = 0
                        meth_keys_buffer = []
                        for island in cpg950_coding_genes[ge_patient[key][0]]:  # Methylation
                            if counter < window_size:
                                if island['site_id'] in methylation_keys:
                                    methylation_vector.append(meth_patient[island['site_id']])
                                    meth_keys_buffer.append(island['site_id'])
                                    counter += 1
                            else:
                                break

                            buffer.append(methylation_vector)
                    else:
                        gene_expression_keys.remove(key)

                if case['last_check']['vital_status'] == 'Dead':  # DEAD cases
                    buffer.append(case['last_check']['days_to_death'])  # Adding label
                else:  # ALIVE cases
                    buffer.append(case['last_check']['days_to_last_followup'])  # Adding label
                dataset.append(buffer)
                break
    return dataset


## MAIN
if __name__ == "__main__":

    # Open LOG file
    logfile = open(LOG_PATH, 'w')
    sys.stdout = DualOutput(sys.stdout, logfile)

    # Loading YAML files
    json_paths = yaml_loader(JSON_PATHS_YAML)
    datastore_paths = yaml_loader(DATASTORE_PATHS_YAML)
    dataset_paths = yaml_loader(DATASET_PATH_YAML)

    # Loading data from JSON datastores
    gene_expression_datastore = json_loader(datastore_paths[GENE_EXPRESSION])
    methylation_datastore = json_loader(datastore_paths[METHYLATION])
    overall_survival_datastore = json_loader(datastore_paths[OVERALL_SURVIVAL])
    cpg950_coding_genes_datastore = json_loader(datastore_paths[CPG950_CODING_GENES])

    gene_expression_keys = [key for key in gene_expression_datastore[0].keys()]
    gene_expression_keys = gene_expression_keys[1:]
    methylation_keys = [key for key in methylation_datastore[0].keys()]
    methylation_keys = methylation_keys[1:]

    i = 0
    dataset = []

    # Use ProcessPoolExecutor with the process_patient function
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(
            process_patient,
            gene_expression_datastore,
            [methylation_datastore] * len(gene_expression_datastore),
            [overall_survival_datastore] * len(gene_expression_datastore),
            [cpg950_coding_genes_datastore] * len(gene_expression_datastore),
            [gene_expression_keys] * len(gene_expression_datastore),
            [methylation_keys] * len(gene_expression_datastore)
        ), total=len(gene_expression_datastore)))

    # Flatten the results and append to the main dataset
    for result in results:
        dataset.extend(result)

    gene_expression_and_methylation_keys = gene_expression_keys + methylation_keys
    print(f"Loaded {i} samples")

    # Storing dataset inside a CSV file
    csv_storer(dataset_paths[GENE_EXPRESSION_WITH_ASSOCIATED_METHYLATION], dataset)
    json_storer(json_paths[GENE_EXPRESSION_WITH_ASSOCIATED_METHYLATION_NAMES], gene_expression_and_methylation_keys)

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()

'''
import numpy as np
import pandas as pd
from tqdm import tqdm
from config.methods.configuration_loader import *
from json_dir.methods.json_loader import *
from json_dir.methods.json_storer import *
from data.methods.csv_storer import *
from logs.methods.log_storer import *


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

    gene_expression_keys = [key for key in gene_expression_datastore[0].keys()]
    gene_expression_keys = gene_expression_keys[1:]
    methylation_keys = [key for key in methylation_datastore[0].keys()]
    methylation_keys = methylation_keys[1:]

    i = 0
    dataset = []
    for ge_patient in gene_expression_datastore:
        for meth_patient in methylation_datastore:
            for case in overall_survival_datastore:
                # Matching Patient for GENE EXPRESSION, METHYLATION and OVERALL SURVIVAL
                if (ge_patient['info']['case_id'] == meth_patient['info']['case_id'] and
                        ge_patient['info']['case_id'] == case['info']['case_id']):
                    buffer = []
                    for key in gene_expression_keys:  # Gene Expression
                        # Checking if TSS value is not NONE and if the considered GENE is in the CPG950 datastore
                        if ge_patient[key][0] in cpg950_coding_genes and ge_patient[key][8] is not None:
                            buffer.append(ge_patient[key][7])  # Adding Gene Expression Value

                            tss_index = ge_patient[key][8]
                            window_size = 100  # Average Methylation Islands per Gene is 95
                            methylation_vector = []

                            counter = 0
                            meth_keys_buffer = []
                            for island in cpg950_coding_genes[ge_patient[key][0]]:  # Methylation
                                if counter < window_size:
                                    if island['site_id'] in methylation_keys:
                                        methylation_vector.append(meth_patient[island['site_id']])
                                        meth_keys_buffer.append(island['site_id'])
                                        counter += 1
                                else:
                                    break

                                buffer.append(methylation_vector)
                        else:
                            gene_expression_keys.remove(key)

                    if case['last_check']['vital_status'] == 'Dead':  # DEAD cases
                        buffer.append(case['last_check']['days_to_death'])  # Adding label
                    else:  # ALIVE cases
                        buffer.append(case['last_check']['days_to_last_followup'])  # Adding label
                    dataset.append(buffer)
                    print(f"Loaded {i} samples")
                    i += 1
                    break
    gene_expression_and_methylation_keys = gene_expression_keys + methylation_keys
    print(f"Loaded {i} samples")

    # Storing dataset inside a CSV file
    csv_storer(dataset_paths[GENE_EXPRESSION_WITH_ASSOCIATED_METHYLATION], dataset)
    json_storer(json_paths[GENE_EXPRESSION_WITH_ASSOCIATED_METHYLATION_NAMES], gene_expression_and_methylation_keys)

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
    '''

'''
    gene_expression_path = 'C:/Users/lucai/Downloads/df_gx_lungs.csv'
    df_fpkm_uq = pd.read_csv(gene_expression_path)


    methylation_path = 'C:/Users/lucai/Downloads/df_meth_lungs.csv'
    df_meth = pd.read_csv(methylation_path)[['ID_site', 'mean', 'median']]
    df_meth = df_meth[df_meth['ID_site'].str.startswith('cg')]


    data_meth = extract_methylation_vectors(df_meth, df_fpkm_uq, cpg950_coding_genes, size_vector=1000, symmetrical_mode=True,
                                            filtro_meth='median', filtro_gene='median')


    df_genes = df_fpkm_uq.merge(data_meth[['gene_id', 'beta_values']], on='gene_id', suffixes=('_df1', '_df2'),
                                how='inner')

    df_genes.to_csv('C:/Users/lucai/Downloads/file.csv', index=False)


    print(data_meth)
'''


'''
def extract_methylation_vectors(methylation_dataframe,
                                gene_expression_dataframe,
                                cpg_data,
                                size_vector,
                                symmetrical_mode,
                                filtro_meth='median',
                                filtro_gene='median'):
    TSS_index = int((131328 / 2) - 1)

    df_meth_val_dict = methylation_dataframe.set_index('ID_site')[filtro_meth].to_dict()

    # Create an empty DataFrame
    data = []

    if not symmetrical_mode:
        end_index_vect = TSS_index
        start_index_vect = int(end_index_vect - size_vector + 1)
    elif symmetrical_mode:
        center_index = TSS_index
        start_index_vect = int(center_index - (size_vector / 2) + 1)
        end_index_vect = int(center_index + (size_vector / 2))

    for i, gene_info in tqdm(gene_expression_dataframe.iterrows(), total=len(gene_expression_dataframe),
                             desc="Estrazione vettori Beta_values"):
        gene_expression = gene_info[filtro_gene]
        vector = [0] * size_vector  # Create a list of zeros
        gene_name = gene_info['gene_name']
        gene_id = gene_info['gene_id']

        if gene_name in cpg_data:
            for island in cpg_data[gene_name]:
                index = int(island['site_index'])
                site_id = island['site_id']

                if start_index_vect <= index <= end_index_vect and site_id in df_meth_val_dict:
                    vector[index - start_index_vect] = df_meth_val_dict[site_id]

        # Append the row to the data list
        data.append([gene_name, gene_id, gene_expression, vector])

    # Convert the data list to a DataFrame
    data_df = pd.DataFrame(data, columns=['gene_name', 'gene_id', 'gene_expression', 'beta_values'])

    return data_df
'''
