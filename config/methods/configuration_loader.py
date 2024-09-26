import yaml
import logging
from error.configuration_file_error import ConfigurationFileError


def yaml_loader(path):
    """
        :param path: YAML file path to load
        :return config: dictionary with data stored from YAML file
    """
    try:
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
            print(f"Data has been correctly loaded from {path} file")
        return config
    except Exception as e:
        logging.error(str(e))
        raise ConfigurationFileError(f"a problem occurred while trying reading {path} file") from None
