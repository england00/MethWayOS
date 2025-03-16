import os
import sys
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from config.methods.configuration_loader import yaml_loader
from json_dir.methods.json_loader import json_loader
from json_dir.methods.json_storer import json_storer
from logs.methods.log_storer import DualOutput


''' 
NOTA: qui viene utilizzato lo  script di Alessandro Monteleone  adattato per restiutire un file  avente per ogni riga un 
      gene ed in  colonna, tra  i vari dati, anche  la lista delle isole  di metilazione presenti  all'interno della suo 
      promotore (che si  trova a monte del TSS nei  geni a strand positivo ed a valle per i geni a  strand negativo). In 
      particolare, la variabile "EXTRACTED_SEQUENCE_DIMENSION" mostra la dimensione della sequenza di ogni gene estratta 
      a tale scopo, incentrata sul TSS indicato tra i dati nel datastore "gene_expression_data.json"
'''


## CONFIGURATION
''' General '''
DATASTORE_PATHS_YAML = '../../config/paths/datastore_paths.yaml'
EXTRACTED_SEQUENCE_DIMENSION = 131328
JSON_PATHS_YAML = '../../config/paths/json_paths.yaml'
LOG_PATH = f'../../logs/files/{os.path.basename(__file__)}.txt'
TABLE_PATHS_YAML = '../../config/paths/table_paths.yaml'

''' Input Data '''
CPG950_CODING_GENES_ORIGINAL = 'cpg950_coding_genes_original'
GENE_EXPRESSION_ALE = 'gene_expression_lungs'
METHYLATION_ALE = 'methylation_lungs'

''' Output Data '''
METHYLATION_VECTORS_FOR_EACH_GENE_FULL = 'methylation_vectors_for_each_gene_full'
SELECTED_METHYLATION_ISLANDS_FULL = 'selected_methylation_islands_full'
MCAT_SELECTED_METHYLATION_ISLANDS = 'MCAT_selected_methylation_islands'


## FUNCTIONS
def extract_methylation_vectors(df_gene_expression, dict_cpg, vector_size):
    tss_index = int((EXTRACTED_SEQUENCE_DIMENSION / 2) - 1)
    list_methylation_vectors = []
    chosen_methylation_islands = defaultdict(list)

    # Indexing Vectors around Sequence Coordinates (based on TSS)
    center_index = tss_index
    start_index = int(center_index - (vector_size / 2) + 1)
    end_index = int(center_index + (vector_size / 2))

    # Extracting Methylation Islands around TSS for each GENE
    for i, gene_info in tqdm(df_gene_expression.iterrows(),
                             total=len(df_gene_expression),
                             desc="Extracting Methylation Islands around TSS"):
        vector = [0] * vector_size

        # Searching 'gene_name' inside the CPG950 dictionary
        if gene_info['gene_name'] in dict_cpg:
            for island in dict_cpg[gene_info['gene_name']]:
                index = int(island['site_index'])

                # Selecting only Methylation Islands in the right position around the TSS
                if gene_info['genomic_strand'] == '+':  # With a POSITIVE STRAND, the promoter is upstream
                    if start_index <= index <= center_index:
                        vector[index - start_index] = island['site_id']
                        chosen_methylation_islands[island['site_id']].append(gene_info['gene_name'])
                elif gene_info['genomic_strand'] == '-':  # With a NEGATIVE STRAND, the promoter is downstream
                    if center_index <= index <= end_index:
                        vector[index - center_index] = island['site_id']
                        chosen_methylation_islands[island['site_id']].append(gene_info['gene_name'])
                '''
                if start_index <= index <= end_index and site_id in dict_values_df_methylation:  # Original operation
                    # vector[index - start_index_vect] = df_meth_val_dict[site_id] 
                    vector[index - start_index] = site_id
                '''

        # Appending each row to the data list
        list_methylation_vectors.append([gene_info['gene_name'], gene_info['gene_id'], vector])

    # Converting the list into a DataFrame
    df_methylation_vectors = pd.DataFrame(list_methylation_vectors, columns=['gene_name', 'gene_id', 'beta_values'])

    return df_methylation_vectors, chosen_methylation_islands


## MAIN
if __name__ == "__main__":
    # Open LOG file
    logfile = open(LOG_PATH, 'w')
    sys.stdout = DualOutput(sys.stdout, logfile)

    # Loading YAML paths
    json_paths = yaml_loader(JSON_PATHS_YAML)
    datastore_paths = yaml_loader(DATASTORE_PATHS_YAML)
    table_paths = yaml_loader(TABLE_PATHS_YAML)

    # Loading data from JSON and CSV paths
    cpg950_coding_genes_dictionary = json_loader(datastore_paths[CPG950_CODING_GENES_ORIGINAL])
    dataframe_gene_expression = pd.read_csv(table_paths[GENE_EXPRESSION_ALE])

    # Extracting Methylation Vectors
    dataframe_methylation_vectors, dictionary_selected_methylation_islands = extract_methylation_vectors(
        dataframe_gene_expression,
        cpg950_coding_genes_dictionary,
        vector_size=1000)

    # Storing data inside a CSV table and a JSON list
    dataframe_methylation_vectors.to_csv(table_paths[METHYLATION_VECTORS_FOR_EACH_GENE_FULL], index=False)
    print(f"Data has been correctly saved inside {table_paths[METHYLATION_VECTORS_FOR_EACH_GENE_FULL]} file")
    json_storer(datastore_paths[SELECTED_METHYLATION_ISLANDS_FULL], dictionary_selected_methylation_islands)

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
