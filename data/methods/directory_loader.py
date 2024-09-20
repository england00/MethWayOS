import os
import logging
from error.general_error import GeneralError


def yaml_loader(path):
    """
        :param path: YAML file path to load
        :return config: dictionary with data stored from YAML file
    """
    try:
        file_nella_cartella = os.listdir(path)

            # Stampa i file
            for file in file_nella_cartella:
                percorso_file = os.path.join(percorso_cartella, file)

                # Verifica che sia un file e non una folder
                if os.path.isfile(percorso_file):
                    print(f"Trovato file: {file}")
    except Exception as e:
        logging.error(str(e))
        raise GeneralError(f"a problem occurred while trying reading {path} file") from None
