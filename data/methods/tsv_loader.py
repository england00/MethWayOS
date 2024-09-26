import csv
import logging
from error.general_error import GeneralError


def tsv_loader(path, dictionary):
    """
        :param path: TSV file path to load
        :param dictionary: data format with 'case_id', 'file_name' and 'file_id' of the current TSV file
        :return buffer: dictionary with all the needed data stored
    """
    try:
        data = []
        with open(path, mode='r', newline='', encoding='utf-8') as file:
            # Reading data
            text = csv.reader(file, delimiter='\t')
            for row in text:
                data.append(row)

            # Formatting data inside a list of dictionaries
            genes_list = data[6:]
            buffer = {'info': dictionary}
            for element in genes_list:
                # Selecting only the coding genes with only 'tpm_unstranded', 'fpkm_unstranded' and	'fpkm_uq_unstranded'
                if element[2] == 'protein_coding':
                    buffer[str(element[0])] = [element[6], element[7], element[8]]

                '''
                # Selecting all the genes with all the information (buffer as a list of dictionaries)
                buffer.append({str(data[1][0]): element[0],  # gene_id
                               str(data[1][1]): element[1],  # gene_name
                               str(data[1][2]): element[2],  # gene_type
                               str(data[1][3]): element[3],  # unstranded
                               str(data[1][4]): element[4],  # stranded_first
                               str(data[1][5]): element[5],  # stranded_second
                               str(data[1][6]): element[6],  # tpm_unstranded
                               str(data[1][7]): element[7],  # fpkm_unstranded
                               str(data[1][8]): element[8]})  # fpkm_uq_unstranded
                '''
            print(f"Data has been correctly loaded from {path} file")
            return buffer
    except FileNotFoundError as e:
        logging.error(str(e))
        raise GeneralError(f"{path} file doesn't exist") from None
