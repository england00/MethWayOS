import csv
import logging
from error.general_error import GeneralError


def csv_storer(path, data, header, keys_mode=False):
    """
        :param path: CSV file to store
        :param data: information to store inside the new CSV file
        :param header: column names
        :param keys_mode: enable/disable keys mode
    """
    try:
        with open(path, mode="w", newline="") as file:
            writer = csv.writer(file, delimiter=',')
            if not keys_mode:
                header.append('y')
                writer.writerow(header)
                writer.writerows(data)
            else:
                writer.writerow(header)
                writer.writerows([[row] for row in data])
            print(f"Data has been correctly saved inside {path} file")
    except OSError as e:
        logging.error(str(e))
        raise GeneralError(f"a problem occurred while trying to open {path} file") from None
