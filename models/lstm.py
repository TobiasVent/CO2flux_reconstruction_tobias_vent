import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class LSTMModelWithStatic(nn.Module):
    def __init__(self, input_size_dyn, input_size_static, hidden_dim, num_layers, dropout):
        super().__init__()
        # LSTM only for dynamic (time-varying) features
        self.lstm = nn.LSTM(
            input_size=input_size_dyn,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)

        # Fully connected layers: combine LSTM output + static context
        combined_dim = hidden_dim + input_size_static
        self.fc1 = nn.Linear(combined_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x_dyn, x_static):
        """
        x_dyn: (batch, seq_len, input_size_dyn)
        x_static: (batch, input_size_static)
        """
        # Pass dynamic input through LSTM
        out, _ = self.lstm(x_dyn)
        last_hidden = out[:, -1, :]           # take final time step
        last_hidden = self.dropout(last_hidden)

        # Concatenate static context
        combined = torch.cat((last_hidden, x_static), dim=1)

        # Final dense layers
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.fc2(x)

        return x.squeeze(-1)


class LSTMModel(nn.Module):
    def __init__(self,input_size, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 1)
        #self.relu = nn.ReLU()
        #self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        x = out[:, -1, :]
        x = self.dropout(x)
        x = self.fc1(x)
        return x.squeeze()
    

class LSTMModelTimeShap(nn.Module):
    def __init__(self,input_size, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 1)
        #self.relu = nn.ReLU()
        #self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        hidden = out[:, -1, :]
        x = self.dropout(hidden)
        x = self.fc1(x)
        return x
    

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_dim,
                        num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)                     # out: (batch, seq_len, hidden_dim)
        x = out[:, -1, :]                        # letzter Zeitschritt
        x = self.dropout(x)
        x = self.fc1(x)
        return x.squeeze()