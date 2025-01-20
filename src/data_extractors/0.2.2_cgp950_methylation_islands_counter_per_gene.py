import os
import sys
import numpy as np
from config.methods.configuration_loader import yaml_loader
from json_dir.methods.json_loader import json_loader
from json_dir.methods.json_storer import json_storer
from logs.methods.log_storer import DualOutput


## CONFIGURATION
''' General '''
DATASTORE_PATHS_YAML = '../../config/paths/datastore_paths.yaml'
JSON_PATHS_YAML = '../../config/paths/json_paths.yaml'
LOG_PATH = f'../../logs/files/{os.path.basename(__file__)}.txt'

''' Output Data '''
CPG950_CODING_GENES_ORIGINAL = 'cpg950_coding_genes_original'


## FUNCTIONS
def counting_elements(data):
    lengths = []
    for key, value in data.items():
        count = len(value)
        lengths.append(count)
        print(f"Key: {key}, Methylation Islands Number: {count}")

    # Calculating statistics
    if lengths:
        mean_length = np.mean(lengths)
        median_length = np.median(lengths)
        max_length = np.max(lengths)
        min_length = np.min(lengths)

        print(f"\nMethylation Islands Lists Lengths Statistics:")
        print(f"Max value: {max_length}")
        print(f"Min value: {min_length}")
        print(f"Mean: {mean_length:.2f}")
        print(f"Median: {median_length}\n")


def sorting_data(data):
    # Based on 'site_index' value
    sorted_data = {}
    for key, value in data.items():
        sorted_data[key] = sorted(value, key=lambda x: x['site_index'])
    return sorted_data


## MAIN
if __name__ == "__main__":

    # Open LOG file
    logfile = open(LOG_PATH, 'w')
    sys.stdout = DualOutput(sys.stdout, logfile)

    # Loading YAML paths
    json_paths = yaml_loader(JSON_PATHS_YAML)
    datastore_paths = yaml_loader(DATASTORE_PATHS_YAML)

    # Storing data from JSON datastore
    cpg950_coding_genes = json_loader(datastore_paths[CPG950_CODING_GENES_ORIGINAL])

    # Counting elements
    counting_elements(cpg950_coding_genes)

    # Sorting data
    cpg950_coding_genes = sorting_data(cpg950_coding_genes)

    # Storing the datastore inside a JSON file
    json_storer(datastore_paths[CPG950_CODING_GENES_ORIGINAL], cpg950_coding_genes)

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
