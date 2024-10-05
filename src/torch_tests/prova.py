import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Configurazione GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Creazione di dati di esempio
X, y = make_classification(n_samples=10000, n_features=200, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizzazione dei dati
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Conversione dei dati in tensori Torch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# Creazione di DataLoader per batch
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)


# Definizione del modello MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Grid Search dei parametri
hidden_sizes = [32, 64, 128]
learning_rates = [0.01, 0.001, 0.0001]
batch_sizes = [16, 32, 64]
num_epochs = 20

best_accuracy = 0.0
best_params = {}

for hidden_size in hidden_sizes:
    for lr in learning_rates:
        for batch_size in batch_sizes:
            # Creazione DataLoader
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            # Creazione e spostamento del modello su GPU
            model = MLP(input_size=X_train.shape[1], hidden_size=hidden_size, output_size=2).to(device)

            # Definizione della funzione di perdita e dell'ottimizzatore
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Addestramento del modello
            for epoch in range(num_epochs):
                model.train()
                for X_batch, y_batch in train_loader:
                    # Forward pass
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)

                    # Backward pass e ottimizzazione
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Valutazione del modello
            model.eval()
            with torch.no_grad():
                outputs = model(X_test_tensor)
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)

                # Salvataggio dei migliori parametri
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {
                        'hidden_size': hidden_size,
                        'learning_rate': lr,
                        'batch_size': batch_size
                    }

            print(
                f'Hidden Size: {hidden_size}, Learning Rate: {lr}, Batch Size: {batch_size}, Accuracy: {accuracy:.4f}')

# Stampa dei migliori parametri
print("\nMigliori parametri trovati:")
print(best_params)
print(f'Migliore accuratezza: {best_accuracy:.4f}')
