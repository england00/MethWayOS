import torch.nn as nn


# Multi-Layer Perceptron
class MLPHidden(nn.Module):
    def __init__(self, input_size, hidden_layer_config, output_size, dropout_rate):
        super(MLPHidden, self).__init__()
        self.hidden_layers_number = len(hidden_layer_config)

        # ONE hidden layer
        if len(hidden_layer_config) == 1:
            self.fc1 = nn.Linear(input_size, hidden_layer_config[0])
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(hidden_layer_config[0], output_size)

        # TWO hidden layers
        elif len(hidden_layer_config) == 2:
            self.fc1 = nn.Linear(input_size, hidden_layer_config[0])
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(hidden_layer_config[0], hidden_layer_config[1])
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout_rate)
            self.fc3 = nn.Linear(hidden_layer_config[1], output_size)

        # THREE hidden layers
        elif len(hidden_layer_config) == 3:
            self.fc1 = nn.Linear(input_size, hidden_layer_config[0])
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(hidden_layer_config[0], hidden_layer_config[1])
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout_rate)
            self.fc3 = nn.Linear(hidden_layer_config[1], hidden_layer_config[2])
            self.relu3 = nn.ReLU()
            self.dropout3 = nn.Dropout(dropout_rate)
            self.fc4 = nn.Linear(hidden_layer_config[2], output_size)

        # FOUR hidden layers
        elif len(hidden_layer_config) == 4:
            self.fc1 = nn.Linear(input_size, hidden_layer_config[0])
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(hidden_layer_config[0], hidden_layer_config[1])
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout_rate)
            self.fc3 = nn.Linear(hidden_layer_config[1], hidden_layer_config[2])
            self.relu3 = nn.ReLU()
            self.dropout3 = nn.Dropout(dropout_rate)
            self.fc4 = nn.Linear(hidden_layer_config[2], hidden_layer_config[3])
            self.relu4 = nn.ReLU()
            self.dropout4 = nn.Dropout(dropout_rate)
            self.fc5 = nn.Linear(hidden_layer_config[3], output_size)

    def forward(self, x):

        # ONE hidden layer
        if self.hidden_layers_number == 1:
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            return x

        # TWO hidden layers
        elif self.hidden_layers_number == 2:
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.dropout2(x)
            x = self.fc3(x)
            return x

        # THREE hidden layers
        elif self.hidden_layers_number == 3:
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.dropout2(x)
            x = self.fc3(x)
            x = self.relu3(x)
            x = self.dropout3(x)
            x = self.fc4(x)
            return x

        # FOUR hidden layers
        elif self.hidden_layers_number == 4:
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.dropout2(x)
            x = self.fc3(x)
            x = self.relu3(x)
            x = self.dropout3(x)
            x = self.fc4(x)
            x = self.relu4(x)
            x = self.dropout4(x)
            x = self.fc5(x)
            return x
