from collections import Counter
from config.methods.configuration_loader import *
from json_dir.methods.json_loader import *

### CONFIGURATION
JSON_PATHS_YAML = '../../config/files/json_paths.yaml'
DIRECTORIES_PATHS_YAML = '../../config/files/directories_paths.yaml'
OVERALL_SURVIVAL = 'overall_survival'


## FUNCTIONS
def myprint(dictionary):
    i = 1
    for element in dictionary:
        print(i, element)
        i += 1


## MAIN
if __name__ == "__main__":

    # Loading YAML files
    json_path = yaml_loader(JSON_PATHS_YAML)
    directories_path = yaml_loader(DIRECTORIES_PATHS_YAML)

    # Storing data from JSON file
    data = json_loader(json_path[OVERALL_SURVIVAL])

    # Storing only information about 'case_id', 'file_name' and 'file_id' for the JSON file
    buffer = []
    for item in data:
        buffer.append(
            {'case_id': item['cases'][0]['case_id'], 'file_name': item['file_name'], 'file_id': item['file_id']})
    data = buffer

    # myprint(data)

    print(directories_path[OVERALL_SURVIVAL])

    import os
    main_path = directories_path[OVERALL_SURVIVAL]


    try:
        subfolders = [file for file in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, file))]

        # Itera attraverso ciascuna folder
        for folder in subfolders:
            percorso_sottocartella = os.path.join(main_path, folder)

            # Ottieni i file nella folder (presupponendo che ci sia un solo file per folder)
            file_nella_sottocartella = os.listdir(percorso_sottocartella)

            # Verifica se c'è almeno un file nella folder
            if file_nella_sottocartella:
                file_path = os.path.join(percorso_sottocartella, file_nella_sottocartella[0])

                # Stampa il nome del file e il percorso completo
                print(f"Trovato file: {file_nella_sottocartella[0]} in {percorso_sottocartella}")

            else:
                print(f"Nessun file trovato in {percorso_sottocartella}")

    except FileNotFoundError:
        print(f"Errore: La cartella {main_path} non esiste.")

