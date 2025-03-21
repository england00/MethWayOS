import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config.methods.configuration_loader import *
from json_dir.methods.json_loader import json_loader
from logs.methods.log_storer import *



## CONFIGURATION
''' General '''
DATASET_PATH_YAML = '../../config/paths/dataset_paths.yaml'
DATASTORE_PATHS_YAML = '../../config/paths/datastore_paths.yaml'
JSON_PATHS_YAML = '../../config/paths/json_paths.yaml'
LOG_PATH = f'../../logs/files/{os.path.basename(__file__)}.txt'
TABLE_PATHS_YAML = '../../config/paths/table_paths.yaml'

''' Input Datastore '''
SELECTED_METHYLATION_ISLANDS_FULL = 'selected_methylation_islands_full'

''' Input Dataset '''
METHYLATION_MCAT = 'methylation450_MCAT'

''' Output Dataset '''
GENE_EXPRESSION_DEEPPROG_methylation = 'gene_expression_DeepProg'
METHYLATION_DEEPPROG = 'methylation_DeepProg'
OVERALL_SURVIVAL_DEEPPROG = 'overall_survival_DeepProg'


## FUNCTIONS
def standard_scale_dataframe(df):
    cols_to_scale = [col for col in df.columns if col != "Samples"]
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    df[cols_to_scale] = df[cols_to_scale].round(4)

    return df


def process_csv(gene_to_cpg, input_file, output_meth):
    # Loading CSV file
    df_meth = pd.read_csv(input_file)
    print(f"Data has been correctly loaded from {input_file} file")

    # Renaming columns
    df_meth.rename(columns={'case_id': 'Samples'}, inplace=True)

    # Using a dictionary for efficient DataFrame creation
    data = {"Samples": df_meth["Samples"]}

    # Selecting columns for CpG sites and computing mean efficiently
    for gene, cpg_sites in gene_to_cpg.items():
        cpg_columns = [cg + "_meth" for cg in cpg_sites if (cg + "_meth") in df_meth.columns]
        if cpg_columns:
            data[gene] = df_meth[cpg_columns].mean(axis=1)

    # Creating DataFrame in one operation
    df_result = pd.DataFrame(data)

    # Scaling CpG sites values
    df_scaled = standard_scale_dataframe(df_result)

    # Storing TSV file of CpG sites
    df_scaled.to_csv(output_meth, sep='\t', index=False)
    print(f"Data has been correctly saved inside {output_meth} file")


def filter_common_samples(df1, df2, df3):
    # 'Samples' intersection
    common_samples = set(df1["Samples"]) & set(df2["Samples"]) & set(df3["Samples"])
    df1_filtered = df1[df1["Samples"].isin(common_samples)]
    df2_filtered = df2[df2["Samples"].isin(common_samples)]
    df3_filtered = df3[df3["Samples"].isin(common_samples)]

    # Alignment
    df1_filtered = df1_filtered.sort_values("Samples").reset_index(drop=True)
    df2_filtered = df2_filtered.sort_values("Samples").reset_index(drop=True)
    df3_filtered = df3_filtered.sort_values("Samples").reset_index(drop=True)

    return df1_filtered, df2_filtered, df3_filtered


## MAIN
if __name__ == "__main__":

    # Open LOG file
    logfile = open(LOG_PATH, 'w')
    sys.stdout = DualOutput(sys.stdout, logfile)

    # Loading YAML paths
    json_paths = yaml_loader(JSON_PATHS_YAML)
    datastore_paths = yaml_loader(DATASTORE_PATHS_YAML)
    dataset_paths = yaml_loader(DATASET_PATH_YAML)

    # Loading JSON file
    promoters_methylation_islands_dictionary = json_loader(datastore_paths[SELECTED_METHYLATION_ISLANDS_FULL])

    # Inverting Genes and CpG sites as keys
    inverted_dict = {}
    for key, values in promoters_methylation_islands_dictionary.items():
        for value in values:
            if value not in inverted_dict:
                inverted_dict[value] = []
            inverted_dict[value].append(key)

    # Processing Datasets data from JSON datastores
    process_csv(inverted_dict,
                dataset_paths[METHYLATION_MCAT],
                dataset_paths[METHYLATION_DEEPPROG])

    # Aligning Datasets
    dataframe_gene_expression = pd.read_csv(dataset_paths[GENE_EXPRESSION_DEEPPROG_methylation], sep="\t")
    dataframe_methylation = pd.read_csv(dataset_paths[METHYLATION_DEEPPROG], sep="\t")
    dataframe_overall_survival = pd.read_csv(dataset_paths[OVERALL_SURVIVAL_DEEPPROG], sep="\t")
    dataframe_gene_expression, dataframe_methylation, dataframe_overall_survival = filter_common_samples(dataframe_gene_expression, dataframe_methylation, dataframe_overall_survival)

    # Storing Datasets
    dataframe_gene_expression.to_csv(dataset_paths[GENE_EXPRESSION_DEEPPROG_methylation], sep='\t', index=False)
    print(f"Data has been correctly saved inside {dataset_paths[GENE_EXPRESSION_DEEPPROG_methylation]} file")
    dataframe_methylation.to_csv(dataset_paths[METHYLATION_DEEPPROG], sep='\t', index=False)
    print(f"Data has been correctly saved inside {dataset_paths[METHYLATION_DEEPPROG]} file")
    dataframe_overall_survival.to_csv(dataset_paths[OVERALL_SURVIVAL_DEEPPROG], sep='\t', index=False)
    print(f"Data has been correctly saved inside {dataset_paths[OVERALL_SURVIVAL_DEEPPROG]} file")

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
