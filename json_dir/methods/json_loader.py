import json
import logging
from error.general_error import GeneralError


def json_loader(path):
    """
        :param path: JSON file path to load
        :return data: dictionary with data stored from JSON file
    """
    try:
        with open(path, 'r') as file:
            data = json.load(file)
            print(f"Data has been correctly loaded from {path} file")
        return data
    except FileNotFoundError as e:
        logging.error(str(e))
        raise GeneralError(f"{path} file doesn't exist") from None
    except json.JSONDecodeError as e:
        logging.error(str(e))
        raise GeneralError(f"{path} is not a valid JSON file") from None
