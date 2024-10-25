import json
import pandas as pd
from tqdm import tqdm
from data.methods.csv_dataset_storer import *



def extract_methylation_vectors(df_meth, df_rna_result, data_cpg, size_vector, symmetrical_mode, filtro_meth='median', filtro_gene='log_median'):
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
        data.append([gene_name,gene_id, gene_expression, vector])

    # Convert the data list to a DataFrame
    data_df = pd.DataFrame(data, columns=['gene_name', 'gene_id', 'gene_expression', 'beta_values'])

    return data_df


## MAIN
if __name__ == "__main__":

    path_gx = '../../data/datasets/df_gx_lungs.csv'
    df_fpkm_uq = pd.read_csv(path_gx)



    with open('../../data/datastore/cpg950_coding_genes_2.json', 'r') as f:
        data_cpg = json.load(f)

    path_meth = '../../data/datasets/df_meth_lungs.csv'
    df_meth = pd.read_csv(path_meth)[['ID_site', 'mean', 'median']]
    df_meth = df_meth[df_meth['ID_site'].str.startswith('cg')]

    data_meth = extract_methylation_vectors(df_meth, df_fpkm_uq, data_cpg, size_vector=1000, symmetrical_mode=True,
                                            filtro_meth='median', filtro_gene='median')

    csv_storer('../../data/datasets/prova.csv', data_meth)
