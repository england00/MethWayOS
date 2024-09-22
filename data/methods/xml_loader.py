import xml.etree.ElementTree as ET
import logging
from error.general_error import *


def namespaces():
    # Managing namespaces inside the file
    spaces = {
        'admin': 'http://tcga.nci/bcr/xml/administration/2.7',
        'brca': 'http://tcga.nci/bcr/xml/clinical/brca/2.7',
        'brca_nte': 'http://tcga.nci/bcr/xml/clinical/brca/shared/new_tumor_event/2.7/1.0',
        'brca_shared': 'http://tcga.nci/bcr/xml/clinical/brca/shared/2.7',
        'clin_shared': 'http://tcga.nci/bcr/xml/clinical/shared/2.7',
        'follow_up_v2.1': 'http://tcga.nci/bcr/xml/clinical/brca/followup/2.7/2.1',
        'follow_up_v4.0': 'http://tcga.nci/bcr/xml/clinical/brca/followup/2.7/4.0',
        'nte': 'http://tcga.nci/bcr/xml/clinical/shared/new_tumor_event/2.7',
        'rad': 'http://tcga.nci/bcr/xml/clinical/radiation/2.7',
        'rx': 'http://tcga.nci/bcr/xml/clinical/pharmaceutical/2.7',
        'shared': 'http://tcga.nci/bcr/xml/shared/2.7',
        'shared_stage': 'http://tcga.nci/bcr/xml/clinical/shared/stage/2.7',
        'xsi': 'http://www.w3.org/2001/XMLSchema-instance',
        'schemaLocation': 'http://tcga.nci/bcr/xml/clinical/brca/2.7'
    }
    return spaces


def followup_versions(spaces, patient):
    versions = ['follow_up_v2.1', 'follow_up_v4.0']
    for item in versions:
        if patient.find(f'brca:follow_ups/{item}:follow_up/clin_shared:vital_status', spaces) is not None:
            return item


def data_store(path, root):
    # Checking if the interested data about the patient are present
    if root.find('brca:patient', namespaces()) is None:
        return None
    else:
        patient_element = root.find('brca:patient', namespaces())

    # Checking the last followup version
    if followup_versions(namespaces(), patient_element) is not None:
        version = followup_versions(namespaces(), patient_element)
        last_check = {
            'vital_status': patient_element.find(f'brca:follow_ups/{version}:follow_up/clin_shared:vital_status',
                                                 namespaces()).text,
            'days_to_last_followup': patient_element.find(
                f'brca:follow_ups/{version}:follow_up/clin_shared:days_to_last_followup', namespaces()).text,
            'days_to_death': patient_element.find(f'brca:follow_ups/{version}:follow_up/clin_shared:days_to_death',
                                                  namespaces()).text}
    else:
        last_check = {'vital_status': patient_element.find('clin_shared:vital_status', namespaces()).text,
                       'days_to_last_followup': patient_element.find('clin_shared:days_to_last_followup',
                                                                     namespaces()).text,
                       'days_to_death': patient_element.find('clin_shared:days_to_death', namespaces()).text}

    return {'info': path, 'last_check': last_check}


def xml_loader(path):
    """
        :param path: XML file path to load
        :return dictionary: all the needed data stored inside a dictionary
    """
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        return data_store(path, root)
    except FileNotFoundError as e:
        logging.error(str(e))
        raise GeneralError(f"{path} file doesn't exist") from None
    except ET.ParseError as e:
        logging.error(str(e))
        raise GeneralError(f"a problem occurred while parsing XML file: {e}") from None
