import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

# Dataset personalizzato
class GeneCpGDataset(Dataset):
    def __init__(self, gene_data, cpg_data, targets):
        self.gene_data = gene_data
        self.cpg_data = cpg_data
        self.targets = targets

    def __len__(self):
        return len(self.gene_data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.gene_data[idx], dtype=torch.float32),
            torch.tensor(self.cpg_data[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )

# Modello Transformer per relazioni Gene-CpG
class GeneCpGTransformer(nn.Module):
    def __init__(self, gene_dim, cpg_dim, hidden_dim=128, num_heads=8, num_layers=2):
        super(GeneCpGTransformer, self).__init__()

        # Embedding per input
        self.gene_embedding = nn.Linear(gene_dim, hidden_dim)
        self.cpg_embedding = nn.Linear(cpg_dim, hidden_dim)

        # Transformer con batch_first=True
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True,  # Usa batch_first per evitare il warning
        )

        # Output layer per predire rilevanza delle CpG
        self.output_layer = nn.Linear(hidden_dim, cpg_dim)

    def forward(self, gene_data, cpg_data):
        # Embedding
        gene_embed = self.gene_embedding(gene_data).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        cpg_embed = self.cpg_embedding(cpg_data).unsqueeze(1)    # (batch_size, 1, hidden_dim)

        # Transformer pass
        transformer_output = self.transformer(gene_embed, cpg_embed)

        # Output prediction
        output = self.output_layer(transformer_output.squeeze(1))  # (batch_size, cpg_dim)
        return output

# Funzione principale
def main():
    # Verifica dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configurazione dei dati
    num_genes = 20000
    num_cpgs = 44000
    num_samples = 100  # Esempio con 100 campioni

    # Simulazione dei dati
    gene_expression_data = np.random.rand(num_samples, num_genes)
    cpg_methylation_data = np.random.rand(num_samples, num_cpgs)
    target_data = np.random.randint(0, 2, (num_samples, num_cpgs))

    # Dataset e DataLoader
    dataset = GeneCpGDataset(gene_expression_data, cpg_methylation_data, target_data)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Configurazione del modello
    gene_dim = num_genes
    cpg_dim = num_cpgs
    model = GeneCpGTransformer(gene_dim, cpg_dim).to(device)

    # Ottimizzatore e loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for gene_data, cpg_data, targets in dataloader:
            # Sposta dati su dispositivo
            gene_data = gene_data.to(device)
            cpg_data = cpg_data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(gene_data, cpg_data)

            # Calcolo della loss
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

    # Valutazione del modello
    model.eval()
    with torch.no_grad():
        for gene_data, cpg_data, targets in dataloader:
            # Sposta dati su dispositivo
            gene_data = gene_data.to(device)
            cpg_data = cpg_data.to(device)

            # Forward pass
            predictions = model(gene_data, cpg_data)
            probs = torch.sigmoid(predictions)  # Converti in probabilità
            print(probs)  # Stampa un esempio delle probabilità

# Esegui lo script
if __name__ == "__main__":
    main()
