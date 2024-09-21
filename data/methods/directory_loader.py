import os
import logging
from error.general_error import GeneralError


def directory_loader(path):
    """
        :param path: main DIRECTORY file path to load
        :return file_paths: list of the file paths stored from the DIRECTORY
    """
    try:
        file_paths = []
        folders = [file for file in os.listdir(path) if os.path.isdir(os.path.join(path, file))]
        for folder in folders:
            folder_path = os.path.join(path, folder)
            file_in_folder = os.listdir(folder_path)  # assuming the presence of only one file per folder
            if file_in_folder:
                file_paths.append(os.path.join(folder_path, file_in_folder[0]))
        return file_paths
    except FileNotFoundError as e:
        logging.error(str(e))
        raise GeneralError(f"{path} file doesn't exist") from None
