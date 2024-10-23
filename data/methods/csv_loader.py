import pyarrow.csv as pv
import logging
from config.methods.configuration_loader import yaml_loader
from json_dir.methods.json_loader import json_loader
from error.general_error import GeneralError


def csv_loader(path, yaml_file, json_file_name, delimiter=';'):
    """
        :param path: CSV file to load
        :param yaml_file: YAML file to load
        :param json_file_name: index inside the JSON file to load
        :param delimiter: settable delimiter character
        :return dataframe: information loaded from the CSV file
        :return dataframe_columns: column names of data loaded from the CSV file
    """
    json_paths = yaml_loader(yaml_file)
    names = json_loader(json_paths[json_file_name])
    if delimiter == ';':
        names.append('y')
    try:
        read_options = pv.ReadOptions(column_names=names,
                                      autogenerate_column_names=False,
                                      block_size=256*1024*1024)  # 256 MB per block
        parse_options = pv.ParseOptions(delimiter=delimiter)
        table = pv.read_csv(input_file=path, read_options=read_options, parse_options=parse_options)
        dataframe = table.to_pandas()
        del table  # releasing memory
        # dataframe = pd.read_csv(path, delimiter=';', header=None, names=names)
        print(f"Data has been correctly loaded from {path} file")
        return dataframe, names
    except FileNotFoundError as e:
        logging.error(str(e))
        raise GeneralError(f"{path} file doesn't exist") from None
    except Exception as e:
        logging.error(str(e))
        raise GeneralError(f"a problem occurred while trying reading {path} file") from None
