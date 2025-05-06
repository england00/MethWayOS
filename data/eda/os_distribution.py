from json_dir.methods.json_loader import json_loader
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Imposta uno stile generale simile a seaborn-whitegrid
plt.style.use('seaborn-v0_8-whitegrid') # Usare una versione specifica per compatibilità

# Colori personalizzati
primary_color = '#66c2a5'
secondary_color = '#fc8d62'
grid_color = '#e0e0e0'
text_color = '#333333'
title_color = '#225ea8' # Un blu più scuro

# Percorso del file
OVERALL_SURVIVAL_PATH = r"D:\Work\Università\Magistrale\6 - Tirocinio\Dati\BRCA\datastores\overall_survival\overall_survival_data.json"
gene_expression_datastore = json_loader(OVERALL_SURVIVAL_PATH)

days_list = []
for record in gene_expression_datastore:
    last_check = record["last_check"]
    status = last_check.get("vital_status")

    if status == "Alive":
        value = last_check.get("days_to_last_followup")
    elif status == "Dead":
        value = last_check.get("days_to_death")
    else:
        value = None  # in caso di valori non attesi

    # Convertiamo a intero solo se esiste e convertiamo in mesi
    days = int(value) if value is not None else None
    months = days / 30.44 if days is not None else None # Media di giorni in un mese
    days_list.append(months)

# Rimuovi i valori None dalla lista per il calcolo della media
valid_months_list = [month for month in days_list if month is not None]

# Creazione del grafico
plt.figure(figsize=(10, 7)) # Aumenta un po' le dimensioni

plt.hist(valid_months_list, bins=400, color=primary_color, edgecolor=secondary_color, alpha=0.7)

# Etichette e titolo con stile
plt.title('', fontsize=18, fontweight='bold', color=title_color, pad=15)
plt.xlabel('Months', fontsize=14, color=text_color, fontweight='semibold')
plt.ylabel('Frequency', fontsize=14, color=text_color, fontweight='semibold')

# Miglioramento grafico con colori personalizzati
plt.grid(True, linestyle='--', alpha=0.6, color=grid_color)
plt.xticks(fontsize=12, color=text_color)
plt.yticks(fontsize=12, color=text_color)

# Aggiunta di una linea orizzontale per evidenziare la media (solo se ci sono dati validi)
if valid_months_list:
    mean_value = np.mean(valid_months_list)
    plt.axvline(mean_value, color=secondary_color, linestyle='--', linewidth=2, label=f'Average: {mean_value:.2f} months')
    plt.legend(fontsize=12, loc='upper right', frameon=False) # Legenda con stile

# Rimozione delle cornici superflue
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Ottimizzazione del layout
plt.tight_layout()

# Salva il grafico
plt.savefig('os_distribution_months.png', dpi=300, bbox_inches='tight')

# Mostra il grafico
plt.show()