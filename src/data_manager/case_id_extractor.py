import json

if __name__ == "__main__":

    # Caricare il file JSON
    with open('file.json', 'r') as f:
        data = json.load(f)

    # Inizializzare un dizionario per immagazzinare i dati
    dati_dict = {}

    # Iterare sui dati e aggiungerli al dizionario
    for item in data:
        file_id = item['file_id']
        dati_dict[file_id] = {
            'data_format': item['data_format'],
            'cases': item['cases'],
            'access': item['access'],
            'file_name': item['file_name'],
            'data_type': item['data_type'],
            'data_category': item['data_category'],
            'experimental_strategy': item['experimental_strategy'],
            'platform': item['platform'],
            'file_size': item['file_size']
        }

    # Stampa del dizionario creato
    print(dati_dict)
