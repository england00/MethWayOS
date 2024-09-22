import csv
import logging
from error.general_error import GeneralError


def csv_storer(path, data):
    """
        :param path: CSV file to store
        :param data: information to store inside the new CSV file
    """
    try:
        with open(path, mode="w", newline="\n") as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerows(data)
            print(f"Data has been correctly saved inside {path} file")
    except OSError as e:
        logging.error(str(e))
        raise GeneralError(f"a problem occurred while trying to open {path} file") from None
