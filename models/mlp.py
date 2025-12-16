import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()


        self.layers = nn.ModuleList()
        prev_dim = input_dim

        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(prev_dim, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)        
            x = self.dropout(x)
        x = self.output_layer(x)
        return x.squeeze()
