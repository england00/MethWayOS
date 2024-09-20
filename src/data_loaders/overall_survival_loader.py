from collections import Counter
from config.methods.configuration_loader import *
from json_dir.methods.json_loader import *

### CONFIGURATION
JSON_PATH_YAML = '../../config/files/json_path.yaml'
OVERALL_SURVIVAL_JSON = 'overall_survival_json'

## MAIN
if __name__ == "__main__":

    # Loading YAML file
    file_path = yaml_loader(JSON_PATH_YAML)

    # Storing data from JSON file
    data = json_loader(file_path[OVERALL_SURVIVAL_JSON])
