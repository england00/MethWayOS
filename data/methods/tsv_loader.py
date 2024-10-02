import csv
import logging
from error.general_error import GeneralError


def tsv_loader(path):
    """
        :param path: TSV file path to load
        :return data: lists of lists with all data stored
    """
    try:
        data = []
        with open(path, mode='r', newline='', encoding='utf-8') as file:
            # Reading data
            text = csv.reader(file, delimiter='\t')
            for row in text:
                data.append(row)
            print(f"Data has been correctly loaded from {path} file")
            return data

    except FileNotFoundError as e:
        logging.error(str(e))
        raise GeneralError(f"{path} file doesn't exist") from None
