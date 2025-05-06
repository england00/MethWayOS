import matplotlib.pyplot as plt
from matplotlib_venn import venn3

# Imposta uno stile generale
plt.style.use('seaborn-whitegrid')

# Colori personalizzati
colors = ['#66c2a5', '#fc8d62', '#8da0cb']

# Crea la figura
plt.figure(figsize=(8,8))

# Crea il Venn diagram con i dati specificati
venn = venn3(
    subsets={
        '100': 77,      # Solo Lista 1
        '010': 128,    # Solo Lista 2
        '001': 107,    # Solo Lista 3
        '110': 297,    # Lista 1 e Lista 2
        '101': 0,      # Lista 1 e Lista 3
        '011': 0,      # Lista 2 e Lista 3
        '111': 743     # Tutte e tre
    },
    set_labels=('Overall Survival', 'Gene Expression', 'Methylation'),
    set_colors=colors,
    alpha=0.7
)

# Personalizza i font e il colore dei numeri
for text in venn.set_labels:
    text.set_fontsize(14)
    text.set_fontweight('bold')

for label in venn.subset_labels:
    if label:
        label.set_fontsize(12)

# Titolo elegante
plt.title("", fontsize=16, fontweight='bold', pad=20)

# Mostra il grafico
plt.tight_layout()
plt.savefig('venn_diagram.png')
plt.show()
