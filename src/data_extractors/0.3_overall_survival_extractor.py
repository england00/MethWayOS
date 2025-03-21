from config.methods.configuration_loader import *
from json_dir.methods.json_loader import *
from json_dir.methods.json_storer import *
from data.methods.directory_loader import *
from data.methods.xml_loader import *
from logs.methods.log_storer import *


## CONFIGURATION
''' General '''
ALIVE_THRESHOLD = 0  # NOTE: 1825 days with MLP, 0 with MCAT
DATASTORE_PATHS_YAML = '../../config/paths/datastore_paths.yaml'
DIRECTORIES_PATHS_YAML = '../../config/paths/directories_paths.yaml'
JSON_PATHS_YAML = '../../config/paths/json_paths.yaml'
LOG_PATH = f'../../logs/files/{os.path.basename(__file__)}.txt'
CANCER_TYPE = 'GBMLGG'   # [BRCA, GBMLGG]

''' Input Data & Output Datastore '''
OVERALL_SURVIVAL = 'overall_survival'


## MAIN
if __name__ == "__main__":

    # Open LOG file
    logfile = open(LOG_PATH, 'w')
    sys.stdout = DualOutput(sys.stdout, logfile)

    # Loading YAML paths
    json_paths = yaml_loader(JSON_PATHS_YAML)
    directories_paths = yaml_loader(DIRECTORIES_PATHS_YAML)
    datastore_paths = yaml_loader(DATASTORE_PATHS_YAML)

    # Loading OVERALL SURVIVAL data from JSON file (only 'case_id', 'file_name' and 'file_id')
    overall_survival_list = json_loader(json_paths[OVERALL_SURVIVAL])
    buffer = []
    for item in overall_survival_list:
        if CANCER_TYPE == 'BRCA':
            buffer.append({'case_id': item['cases'][0]['case_id'], 'file_name': item['file_name'], 'file_id': item['file_id']})
        elif CANCER_TYPE == 'GBMLGG':
            buffer.append({'case_id': item['associated_entities'][0]['case_id'], 'file_name': item['file_name'], 'file_id': item['file_id']})
    overall_survival_list = buffer

    # Loading all the XML paths
    file_paths = directory_loader(directories_paths[OVERALL_SURVIVAL])
    i = 0
    overall_survival_datastore = []
    for path in file_paths:
        if xml_loader(path, cancer_type=CANCER_TYPE) is not None:
            i += 1
            overall_survival_datastore.append(xml_loader(path, cancer_type=CANCER_TYPE))
    print(f"Loaded {i} paths")

    # Selecting only DEAD cases
    buffer = []
    for dictionary in overall_survival_datastore:
        if ((dictionary['last_check']['days_to_last_followup'] is not None or dictionary['last_check']['days_to_death'] is not None)
            and (dictionary['last_check']['vital_status'] == 'Dead'                                     # DEAD
            or (dictionary['last_check']['vital_status'] == 'Alive'
            and int(dictionary['last_check']['days_to_last_followup']) >= ALIVE_THRESHOLD))):           # ALIVE
            buffer.append(dictionary)
            # Adding for each dictionary 'case_id', 'file_name' and 'file_id'
            file_name = dictionary['info'].split('/')[len(dictionary['info'].split('/')) - 1]
            for item in overall_survival_list:
                if item['file_name'] == file_name:
                    dictionary['info'] = item
                    break
    overall_survival_datastore = buffer

    # Removing duplicates, taking only one case_id for each-one
    buffer = []
    case_ids = []
    for dictionary in overall_survival_datastore:
        if dictionary['info']['case_id'] not in case_ids:
            buffer.append(dictionary)
            case_ids.append(dictionary['info']['case_id'])
        elif dictionary['info']['case_id'] in case_ids and dictionary['last_check']['vital_status'] == 'Dead':
            for element in buffer:
                if element['info']['case_id'] == dictionary['info']['case_id']:
                    if element['last_check']["vital_status"] == 'Alive':
                        element = dictionary
                    break
    gene_expression_list = buffer

    # Storing the datastore inside a JSON file
    json_storer(datastore_paths[OVERALL_SURVIVAL], overall_survival_datastore)

    # Close LOG file
    sys.stdout = sys.__stdout__
    logfile.close()
