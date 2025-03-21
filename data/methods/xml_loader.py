import xml.etree.ElementTree as ET
import logging
from error.general_error import *


## BRCA
def namespaces_brca():
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


def followup_versions_brca(spaces, patient):
    versions = ['follow_up_v2.1', 'follow_up_v4.0']
    for item in versions:
        if patient.find(f'brca:follow_ups/{item}:follow_up/clin_shared:vital_status', spaces) is not None:
            return item


def data_store_brca(path, root):
    # Checking if the interested data about the patient are present
    if root.find('brca:patient', namespaces_brca()) is None:
        return None
    else:
        patient_element = root.find('brca:patient', namespaces_brca())

    # Checking the last followup version
    if followup_versions_brca(namespaces_brca(), patient_element) is not None:
        version = followup_versions_brca(namespaces_brca(), patient_element)
        last_check = {
            'vital_status': patient_element.find(f'brca:follow_ups/{version}:follow_up/clin_shared:vital_status',
                                                 namespaces_brca()).text,
            'days_to_last_followup': patient_element.find(
                f'brca:follow_ups/{version}:follow_up/clin_shared:days_to_last_followup', namespaces_brca()).text,
            'days_to_death': patient_element.find(f'brca:follow_ups/{version}:follow_up/clin_shared:days_to_death',
                                                  namespaces_brca()).text}
    else:
        last_check = {'vital_status': patient_element.find('clin_shared:vital_status', namespaces_brca()).text,
                       'days_to_last_followup': patient_element.find('clin_shared:days_to_last_followup',
                                                                     namespaces_brca()).text,
                       'days_to_death': patient_element.find('clin_shared:days_to_death', namespaces_brca()).text}

    return {'info': path, 'last_check': last_check}


## GBM & LGG
def detect_namespace_gbmlgg(root):
    if root.tag.startswith("{http://tcga.nci/bcr/xml/clinical/gbm/"):
        return "gbm"
    elif root.tag.startswith("{http://tcga.nci/bcr/xml/clinical/lgg/"):
        return "lgg"
    else:
        raise GeneralError("Unsupported XML format")


def namespaces_gbmlgg(ns):
    base_url = f"http://tcga.nci/bcr/xml/clinical/{ns}/2.7"
    return {
        'admin': 'http://tcga.nci/bcr/xml/administration/2.7',
        ns: base_url,
        f'{ns}_nte': f'{base_url}/shared/new_tumor_event/2.7/1.0',
        'clin_shared': 'http://tcga.nci/bcr/xml/clinical/shared/2.7',
        'nte': 'http://tcga.nci/bcr/xml/clinical/shared/new_tumor_event/2.7',
        'rad': 'http://tcga.nci/bcr/xml/clinical/radiation/2.7',
        'rx': 'http://tcga.nci/bcr/xml/clinical/pharmaceutical/2.7',
        'shared': 'http://tcga.nci/bcr/xml/shared/2.7',
        'follow_up_v1.0': f'{base_url}/followup/2.7/1.0',
        'xsi': 'http://www.w3.org/2001/XMLSchema-instance',
    }


def followup_versions_gbmlgg(spaces, patient, ns):
    versions = ['follow_up_v1.0']
    for item in versions:
        if patient.find(f'{ns}:follow_ups/{item}:follow_up/clin_shared:vital_status', spaces) is not None:
            return item


def data_store_gbmlgg(path, root, ns):
    if root.find(f'{ns}:patient', namespaces_gbmlgg(ns)) is None:
        return None
    patient_element = root.find(f'{ns}:patient', namespaces_gbmlgg(ns))
    version = followup_versions_gbmlgg(namespaces_gbmlgg(ns), patient_element, ns)

    if version:
        last_check = {
            'vital_status': patient_element.find(f'{ns}:follow_ups/{version}:follow_up/clin_shared:vital_status',
                                                 namespaces_gbmlgg(ns)).text,
            'days_to_last_followup': patient_element.find(
                f'{ns}:follow_ups/{version}:follow_up/clin_shared:days_to_last_followup', namespaces_gbmlgg(ns)).text,
            'days_to_death': patient_element.find(f'{ns}:follow_ups/{version}:follow_up/clin_shared:days_to_death',
                                                  namespaces_gbmlgg(ns)).text,
        }
    else:
        last_check = {
            'vital_status': patient_element.find('clin_shared:vital_status', namespaces_gbmlgg(ns)).text,
            'days_to_last_followup': patient_element.find('clin_shared:days_to_last_followup', namespaces_gbmlgg(ns)).text,
            'days_to_death': patient_element.find('clin_shared:days_to_death', namespaces_gbmlgg(ns)).text,
        }

    return {'info': path, 'last_check': last_check}


def xml_loader(path, cancer_type: str = "BRCA"):
    """
        :param cancer_type:
        :param path: XML file path to load
        :return dictionary: all the needed data stored inside a dictionary
    """
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        if cancer_type == "BRCA":
            print(f"Data has been correctly loaded from {path} file")
            return data_store_brca(path, root)
        elif cancer_type == "GBMLGG":
            ns = detect_namespace_gbmlgg(root)
            print(f"Data has been correctly loaded from {path} file")
            return data_store_gbmlgg(path, root, ns)
    except FileNotFoundError as e:
        logging.error(str(e))
        raise GeneralError(f"{path} file doesn't exist") from None
    except ET.ParseError as e:
        logging.error(str(e))
        raise GeneralError(f"A problem occurred while parsing XML file: {e}") from None
