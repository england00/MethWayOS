import json
import logging
from error.general_error import GeneralError


def json_storer(path, data):
    """
        :param path: JSON file path to store
        :param data: information to serialize in JSON format
    """
    try:
        with open(path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
            print(f"Data has been correctly saved inside {path} file")
    except OSError as e:
        logging.error(str(e))
        raise GeneralError(f"a problem occurred while trying to open {path} file") from None
    except (TypeError, ValueError) as e:
        logging.error(str(e))
        raise GeneralError(f"a problem occurred while serializing data in JSON format") from None
