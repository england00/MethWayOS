import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Imposta uno stile generale simile a seaborn-whitegrid
plt.style.use('seaborn-v0_8-whitegrid') # Usare una versione specifica per compatibilità

# Colori personalizzati
primary_color = '#1f77b4'  # Blu scuro (colore standard di matplotlib)
censored_color = '#fc8d62'
grid_color = '#e0e0e0'
text_color = '#333333'
title_color = '#225ea8' # Un blu più scuro

# Percorso del file
SURVIVAL = r"D:\Work\Università\Magistrale\6 - Tirocinio\Dati\BRCA\datasets\BRCA_MCAT_methylation450_and_overall_survival_dataset.csv"
df = pd.read_csv(SURVIVAL)
survival_months_list = df['survival_months'].tolist()
censorship_list = df['censorship'].tolist()

# Converti le liste in array numpy per un'indicizzazione più semplice
survival_months_array = np.array(survival_months_list)
censorship_array = np.array(censorship_list)

# Identifica gli indici dove la censura è 1.0
censored_indices = np.where(censorship_array == 1.0)[0]

# Separa i dati censurati e non censurati
censored_survival_months = survival_months_array[censored_indices]
non_censored_survival_months = np.delete(survival_months_array, censored_indices)

# Rimuovi i valori NaN da entrambe le liste
censored_survival_months = censored_survival_months[~np.isnan(censored_survival_months)]
non_censored_survival_months = non_censored_survival_months[~np.isnan(non_censored_survival_months)]

# Calcola la media di tutti i valori di sopravvivenza (prima della separazione)
valid_survival_months = survival_months_array[~np.isnan(survival_months_array)]
mean_value = np.mean(valid_survival_months) if valid_survival_months.size > 0 else None

# Creazione del grafico
plt.figure(figsize=(10, 7)) # Aumenta un po' le dimensioni

# Crea due istogrammi separati con colori diversi
plt.hist(non_censored_survival_months, bins=400, color=primary_color, alpha=0.7, label='Non-Censored')
plt.hist(censored_survival_months, bins=400, color=censored_color, alpha=0.7, label='Censored')

# Aggiunta della linea orizzontale per evidenziare la media (solo se ci sono dati validi)
if mean_value is not None:
    plt.axvline(mean_value, color='gray', linestyle='--', linewidth=2, label=f'Average: {mean_value:.2f}')

# Etichette e titolo con stile
plt.title('', fontsize=18, fontweight='bold', color=title_color, pad=15)
plt.xlabel('Months', fontsize=14, color=text_color, fontweight='semibold')
plt.ylabel('Frequency', fontsize=14, color=text_color, fontweight='semibold')

# Miglioramento grafico con colori personalizzati
plt.grid(True, linestyle='--', alpha=0.6, color=grid_color)
plt.xticks(fontsize=12, color=text_color)
plt.yticks(fontsize=12, color=text_color)

# Aggiunta della legenda
plt.legend(fontsize=12, loc='upper right', frameon=False)

# Rimozione delle cornici superflue
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Ottimizzazione del layout
plt.tight_layout()

# Salva il grafico
plt.savefig('os_distribution_censored_average.png', dpi=300, bbox_inches='tight')

# Mostra il grafico
plt.show()