import pandas as pd
from config.methods.configuration_loader import *
from data.methods.csv_dataset_loader import csv_loader
from data.methods.directory_loader import *
from json_dir.methods.json_loader import *
from logs.methods.log_storer import *


## CONFIGURATION
''' General '''
DATASTORE_PATHS_YAML = '../../config/paths/datastore_paths.yaml'
JSON_PATHS_YAML = '../../config/paths/json_paths.yaml'
LOG_PATH = f'../../logs/files/{os.path.basename(__file__)}.txt'
TABLE_PATHS_YAML = '../../config/paths/table_paths.yaml'
MODE = 'full'   # ['full', 'promoters']
TYPE = '450' # ['27', '27full', '450', '450full']

############## [        'full'       ,     'promoters'    ]
# [    '27'    ,         1478        ,        817        ]
# [            , 1.6400541271989175  , 1.0514075887392902]
# [  '27full'  ,         2680        ,        817        ]
# [            , 4.6138059701492535  , 1.0514075887392902]
# [    '450'   ,         2559        ,        2176       ]
# [            , 6.41305197342712    , 2.4099264705882355]
# [  '450full' ,         2909        ,        2176       ]
# [            , 55.17875558611207   , 2.4099264705882355]

''' Input Datastore '''
CPG950_CODING_GENES = 'cpg950_coding_genes'
SELECTED_METHYLATION_ISLANDS_FULL = 'selected_methylation_islands_full'

''' Input Keys'''
METHYLATION_27_KEYS = 'methylation27_keys'
METHYLATION_27_FULL_KEYS = 'methylation27_full_keys'
METHYLATION_450_KEYS = 'methylation450_keys'
METHYLATION_450_FULL_KEYS = 'methylation450_full_keys'
MCAT_SIGNATURES = 'mcat_signatures'
SURVPATH_SIGNATURES = 'survpath_signatures'

''' Output Keys '''
METHYLATION_SIGNATURES_PROMOTERS_27 = 'methylation_signatures_promoters_27'
METHYLATION_SIGNATURES_PROMOTERS_27_FULL = 'methylation_signatures_promoters_27_full'
METHYLATION_SIGNATURES_PROMOTERS_450 = 'methylation_signatures_promoters_450'
METHYLATION_SIGNATURES_PROMOTERS_450_FULL = 'methylation_signatures_promoters_450_full'
METHYLATION_SIGNATURES_FULL_27 = 'methylation_signatures_full_27'
METHYLATION_SIGNATURES_FULL_27_FULL = 'methylation_signatures_full_27_full'
METHYLATION_SIGNATURES_FULL_450 = 'methylation_signatures_full_450'
METHYLATION_SIGNATURES_FULL_450_FULL = 'methylation_signatures_full_450_full'


## MAIN
if __name__ == "__main__":

    # Open LOG file
    logfile = open(LOG_PATH, 'w')
    sys.stdout = DualOutput(sys.stdout, logfile)

    # Loading YAML paths
    json_paths = yaml_loader(JSON_PATHS_YAML)
    datastore_paths = yaml_loader(DATASTORE_PATHS_YAML)
    table_paths = yaml_loader(TABLE_PATHS_YAML)

    # Loading data from CSV and JSON files
    cpg950_methylation_islands_dictionary = json_loader(datastore_paths[CPG950_CODING_GENES])
    promoters_methylation_islands_dictionary = json_loader(datastore_paths[SELECTED_METHYLATION_ISLANDS_FULL])
    if TYPE == '27':
        general_methylation_islands_dataframe, _ = csv_loader(table_paths[METHYLATION_27_KEYS])
    elif TYPE == '27full':
        general_methylation_islands_dataframe, _ = csv_loader(table_paths[METHYLATION_27_FULL_KEYS])
    if TYPE == '450':
        general_methylation_islands_dataframe, _ = csv_loader(table_paths[METHYLATION_450_KEYS])
    elif TYPE == '450full':
        general_methylation_islands_dataframe, _ = csv_loader(table_paths[METHYLATION_450_FULL_KEYS])
    gene_expression_list_dataframe, _ = csv_loader(table_paths[MCAT_SIGNATURES])

    # Changing format of CPG950 Sites Dictionary
    if MODE == 'full':
        for key, value in cpg950_methylation_islands_dictionary.items():
            cpg950_methylation_islands_dictionary[key] = cpg950_methylation_islands_dictionary[key]['keys']
        promoters_methylation_islands_dictionary = cpg950_methylation_islands_dictionary

    # Promoter Sites Dictionary Selection
    buffer = {}
    valid_methylation_sites = set(general_methylation_islands_dataframe['methylation_island'])
    for key, value in promoters_methylation_islands_dictionary.items():
        if key in valid_methylation_sites:
            buffer[key] = value
    promoters_methylation_islands_dictionary = buffer

    # Promoter Sites Dictionary Inversion
    genes_methylation_sites_dictionary = {}
    for methylation_site, genes in promoters_methylation_islands_dictionary.items():
        for gene in genes:
            if gene not in genes_methylation_sites_dictionary:
                genes_methylation_sites_dictionary[gene] = []
            genes_methylation_sites_dictionary[gene].append(methylation_site)

    # Genes Dictionary Selection
    genes = []
    for column in gene_expression_list_dataframe.columns:
        for gene in gene_expression_list_dataframe[column]:
            if gene != '':
                genes.append(gene)
    buffer = {}
    for key, value in genes_methylation_sites_dictionary.items():
        if key in genes:
            buffer[key] = value
    genes_methylation_sites_dictionary = buffer

    # Storing dataset inside a CSV file
    methylation_dataframe = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in genes_methylation_sites_dictionary.items()]))
    if MODE == 'promoters':
        if TYPE == '27':
            methylation_dataframe.to_csv(table_paths[METHYLATION_SIGNATURES_PROMOTERS_27], index=False)
        elif TYPE == '27full':
            methylation_dataframe.to_csv(table_paths[METHYLATION_SIGNATURES_PROMOTERS_27_FULL], index=False)
        if TYPE == '450':
            methylation_dataframe.to_csv(table_paths[METHYLATION_SIGNATURES_PROMOTERS_450], index=False)
        elif TYPE == '450full':
            methylation_dataframe.to_csv(table_paths[METHYLATION_SIGNATURES_PROMOTERS_450_FULL], index=False)
    elif MODE == 'full':
        if TYPE == '27':
            methylation_dataframe.to_csv(table_paths[METHYLATION_SIGNATURES_FULL_27], index=False)
        elif TYPE == '27full':
            methylation_dataframe.to_csv(table_paths[METHYLATION_SIGNATURES_FULL_27_FULL], index=False)
        if TYPE == '450':
            methylation_dataframe.to_csv(table_paths[METHYLATION_SIGNATURES_FULL_450], index=False)
        elif TYPE == '450full':
            methylation_dataframe.to_csv(table_paths[METHYLATION_SIGNATURES_FULL_450_FULL], index=False)

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
