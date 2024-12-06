import pyarrow.csv as pv
import logging
from error.general_error import GeneralError


def csv_loader(path):
    """
        :param path: CSV file to load
        :return dataframe: information loaded from the CSV file
        :return dataframe_columns: column names of data loaded from the CSV file
    """
    try:
        # Reading columns names
        with open(path, mode="r") as file:
            first_line = file.readline().strip()
            names = first_line.split(",")

        # Reading data
        read_options = pv.ReadOptions(column_names=names,
                                      autogenerate_column_names=False,
                                      skip_rows=1,
                                      block_size=256*1024*1024)  # 256 MB per block
        parse_options = pv.ParseOptions(delimiter=',')
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
