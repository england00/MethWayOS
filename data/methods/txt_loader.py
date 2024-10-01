import logging
from error.general_error import GeneralError


def txt_loader(path):
    """
        :param path: TXT file to load
        :return features: features list read from the TXT file
        :return values: values list read from the TXT file
    """
    try:
        features = values = []
        with open(path, 'r') as file:
            for line in file:
                columns = line.strip().split('\t')
                features.append(columns[0])
                values.append(columns[1] if columns[1] != 'NA' else None)
        print(f"Data has been correctly loaded from {path} file")
        return features, values
    except FileNotFoundError as e:
        logging.error(str(e))
        raise GeneralError(f"{path} file doesn't exist") from None
    except Exception as e:
        logging.error(str(e))
        raise GeneralError(f"a problem occurred while trying reading {path} file") from None
