from config.methods.configuration_loader import *
from config.methods.json_loader import *

### CONFIGURATION
JSON_PATH_YAML = '../../config/files/json_path.yaml'
GENE_ESPRESSION_JSON = 'gene_expression_json'
METHYLATION_JSON = 'methylation_json'
OVERALL_SURVIVAL = 'overall_survival_json'


## FUNCTIONS
def myprint(dictionary):
    i = 1
    for element in dictionary:
        print(i, element)
        i += 1


## MAIN
if __name__ == "__main__":

    data = {}
    file_path = yaml_loader(JSON_PATH_YAML)
    data[GENE_ESPRESSION_JSON] = json_loader(file_path[GENE_ESPRESSION_JSON])
    data[METHYLATION_JSON] = json_loader(file_path[METHYLATION_JSON])
    data[OVERALL_SURVIVAL] = json_loader(file_path[OVERALL_SURVIVAL])

    # Storing only information about 'case_id', 'file_name' and 'file_id' for each JSON file
    for file in data:
        buffer = []
        for item in data[file]:
            buffer.append(
                {'case_id': item['cases'][0]['case_id'], 'file_name': item['file_name'], 'file_id': item['file_id']})
        data[file] = buffer

    '''
    # FIRST CASE: Overall Survival based only on Gene Expression
    overall_survival_ids = {record['case_id'] for record in data['overall_survival_json']}



    # Filtra i record in gene_expression_json che hanno un case_id presente in overall_survival_json
    filtered_gene_expression = [record for record in data['gene_expression_json'] if record['case_id'] in overall_survival_ids]

    
        '''
