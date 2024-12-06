import csv
import logging
from error.general_error import GeneralError


def csv_storer(path, data, header):
    """
        :param path: CSV file to store
        :param data: information to store inside the new CSV file
        :param header: column names
    """
    try:
        with open(path, mode="w", newline="") as file:
            writer = csv.writer(file, delimiter=',')
            header.append('y')
            writer.writerow(header)
            writer.writerows(data)
            print(f"Data has been correctly saved inside {path} file")
    except OSError as e:
        logging.error(str(e))
        raise GeneralError(f"a problem occurred while trying to open {path} file") from None
