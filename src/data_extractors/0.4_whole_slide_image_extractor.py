from config.methods.configuration_loader import *
from json_dir.methods.json_loader import *
from json_dir.methods.json_storer import *
from data.methods.directory_loader import *
from logs.methods.log_storer import *


## CONFIGURATION
ALIVE_THRESHOLD = 1825  # NOTE: 1825 days with MLP, 0 with MCAT
DATASTORE_PATHS_YAML = '../../config/paths/datastore_paths.yaml'
DIRECTORIES_PATHS_YAML = '../../config/paths/directories_paths.yaml'
JSON_PATHS_YAML = '../../config/paths/json_paths.yaml'
LOG_PATH = f'../../logs/files/{os.path.basename(__file__)}.txt'
WHOLE_SLIDE_IMAGE = 'whole_slide_image'


## MAIN
if __name__ == "__main__":

    # Open LOG file
    logfile = open(LOG_PATH, 'w')
    sys.stdout = DualOutput(sys.stdout, logfile)

    # Loading YAML paths
    json_paths = yaml_loader(JSON_PATHS_YAML)
    directories_paths = yaml_loader(DIRECTORIES_PATHS_YAML)
    datastore_paths = yaml_loader(DATASTORE_PATHS_YAML)

    # Loading WHOLE SLIDE IMAGE data from JSON file
    whole_slide_image_list = json_loader(json_paths[WHOLE_SLIDE_IMAGE])

    # Removing duplicates, taking only one case_id for each-one
    unique_cases = {}
    for entry in whole_slide_image_list:
        case_id = entry['associated_entities'][0]['case_id']
        if case_id not in unique_cases:
            unique_cases[case_id] = {'slide_id': entry['file_name'],
                                     'patient': '-'.join(entry['associated_entities'][0]['entity_submitter_id'].split('-')[:3])}

    # Storing the datastore inside a JSON file
    json_storer(datastore_paths[WHOLE_SLIDE_IMAGE], unique_cases)

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
