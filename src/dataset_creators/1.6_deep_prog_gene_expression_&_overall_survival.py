import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config.methods.configuration_loader import *
from logs.methods.log_storer import *



## CONFIGURATION
''' General '''
DATASET_PATH_YAML = '../../config/paths/dataset_paths.yaml'
JSON_PATHS_YAML = '../../config/paths/json_paths.yaml'
LOG_PATH = f'../../logs/files/{os.path.basename(__file__)}.txt'
TABLE_PATHS_YAML = '../../config/paths/table_paths.yaml'

''' Input Dataset '''
GENE_EXPRESSION_MCAT_methylation = 'gene_expression_MCAT_methylation'

''' Output Dataset '''
GENE_EXPRESSION_DEEPPROG_methylation = 'gene_expression_DeepProg'
OVERALL_SURVIVAL_DEEPPROG = 'overall_survival_DeepProg'


## FUNCTIONS
def standard_scale_dataframe(df):
    cols_to_scale = [col for col in df.columns if col != "Samples"]
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    df[cols_to_scale] = df[cols_to_scale].round(4)

    return df


def process_csv(input_file, output_rnaseq, output_surv):
    # Loading CSV file
    df = pd.read_csv(input_file)
    print(f"Data has been correctly loaded from {input_file} file")

    # Selecting columns for RNA-Seq
    selected_cols = ['case_id'] + [col for col in df.columns if col.endswith('_rnaseq')]
    df_rnaseq = df[selected_cols]

    # Renaming columns
    df_rnaseq = df_rnaseq.rename(columns={'case_id': 'Samples'})
    df_rnaseq.columns = [col.replace('_rnaseq', '') for col in df_rnaseq.columns]

    # Scaling RNA-Seq values
    df_scaled = standard_scale_dataframe(df_rnaseq)

    # Storing TSV file of RNA-Seq
    df_scaled.to_csv(output_rnaseq, sep='\t', index=False)
    print(f"Data has been correctly saved inside {output_rnaseq} file")

    # Selecting columns for Survival
    df_surv = df[['case_id', 'survival_months', 'censorship']].copy()
    df_surv['survival_months'] = (df_surv['survival_months'] * 30).astype(int)

    # Renaming columns
    df_surv = df_surv.rename(columns={'case_id': 'Samples', 'survival_months': 'days', 'censorship': 'event'})

    # Storing TSV file of Survival
    df_surv.to_csv(output_surv, sep='\t', index=False)
    print(f"Data has been correctly saved inside {output_surv} file")


## MAIN
if __name__ == "__main__":

    # Open LOG file
    logfile = open(LOG_PATH, 'w')
    sys.stdout = DualOutput(sys.stdout, logfile)

    # Loading YAML paths
    json_paths = yaml_loader(JSON_PATHS_YAML)
    dataset_paths = yaml_loader(DATASET_PATH_YAML)

    # Processing Datasets data from JSON datastores
    process_csv(dataset_paths[GENE_EXPRESSION_MCAT_methylation],
                dataset_paths[GENE_EXPRESSION_DEEPPROG_methylation],
                dataset_paths[OVERALL_SURVIVAL_DEEPPROG])

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
