import torch
import torch.nn as nn
import torch.nn.functional as F




class InputAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(InputAttention, self).__init__()
        # These are the weight matrices (W_epsilon, U_epsilon, z_epsilon) from the paper
        self.W = nn.Linear(hidden_size * 2, hidden_size)  # For [h_prev; c_prev]
        self.U = nn.Linear(input_size, hidden_size)       # For x_t
        self.V = nn.Linear(hidden_size, input_size)

    def forward(self, x, h_prev, c_prev):
        # Concatenate the previous hidden and cell states
        combined_state = torch.cat([h_prev, c_prev], dim=1)
        
        # Calculate the attention scores (epsilon_t^k from the paper)
        # The equation is simplified here for clarity and efficient tensor operations
        attention_scores = self.V(torch.tanh(self.W(combined_state) + self.U(x)))
        
        # Apply softmax to get attention weights (alpha_t^k from the paper)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply the attention weights to the input features
        weighted_x = attention_weights * x
        
        return weighted_x, attention_weights

# ================================================
# MODIFIED: LSTM Model with Attention
# The forward pass now manually loops through the sequence
# ================================================
class LSTMModelAttention(nn.Module):
    def __init__(self, input_size, hidden_dim, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        
        
        # We use a single LSTMCell instead of nn.LSTM
        self.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_dim)
        
        # Initialize the attention module
        self.attention = InputAttention(input_size=input_size, hidden_size=hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, seq_len, num_features = x.size()
        
        # Initialize hidden and cell states for the first time step
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        attention_wheigts_list =[]
        # Loop through the sequence time steps
        for t in range(seq_len):
            # Extract the input for the current time step
            x_t = x[:, t, :]
            
            # Apply the attention mechanism to get a weighted input
            weighted_x, attention_wheigts = self.attention(x_t, h_t, c_t)
            
            # Pass the weighted input to the LSTM cell
            h_t, c_t = self.lstm_cell(weighted_x, (h_t, c_t))
            attention_wheigts_list.append(attention_wheigts)
        # The output of the LSTM is the final hidden state
        x = h_t
        
        x = self.dropout(x)
        x = self.fc1(x)
        
        return x.squeeze(),attention_wheigts_list
    

class LSTMModelAttentionTemporal(nn.Module):
    def __init__(self, input_size, hidden_dim, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_dim)
        self.attention = InputAttention(input_size=input_size, hidden_size=hidden_dim)
        self.lstm_layer = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.temporal_attention = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 1)


    def forward(self, x):
        batch_size, seq_len, num_features = x.size()
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        hidden_states = []
        attention_weights_list_input = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            weighted_x, attention_weights = self.attention(x_t, h_t, c_t)
            h_t, c_t = self.lstm_cell(weighted_x, (h_t, c_t))
            hidden_states.append(h_t.unsqueeze(1))  # (batch, 1, hidden_dim)
            attention_weights_list_input.append(attention_weights)
        hidden_states = torch.cat(hidden_states, dim=1)  # (batch, seq_len, hidden_dim)
        attn_scores = self.temporal_attention(hidden_states)  # (batch, seq_len, 1)
        attn_weights_temp = torch.softmax(attn_scores, dim=1)      # (batch, seq_len, 1)
        context = torch.sum(attn_weights_temp * hidden_states, dim=1)
        #weighted_hidden = attn_weights * hidden_states
        #x = self.lstm_layer(weighted_hidden)  

        x = self.dropout(context)
        x = self.fc1(x)
        #return x.squeeze(), attn_weights.squeeze(-1)  # Attention-Gewichte für Analyse
        return x.squeeze()



class LSTMModelAttentionTemporalWithWeights(nn.Module):
    def __init__(self, input_size, hidden_dim, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_dim)
        self.attention = InputAttention(input_size=input_size, hidden_size=hidden_dim)
        self.lstm_layer = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.temporal_attention = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 1)


    def forward(self, x):
        batch_size, seq_len, num_features = x.size()
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        hidden_states = []
        attention_weights_list_input = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            weighted_x, attention_weights = self.attention(x_t, h_t, c_t)
            h_t, c_t = self.lstm_cell(weighted_x, (h_t, c_t))
            hidden_states.append(h_t.unsqueeze(1))  # (batch, 1, hidden_dim)
            attention_weights_list_input.append(attention_weights)
        hidden_states = torch.cat(hidden_states, dim=1)  # (batch, seq_len, hidden_dim)
        attn_scores = self.temporal_attention(hidden_states)  # (batch, seq_len, 1)
        attn_weights_temp = torch.softmax(attn_scores, dim=1)      # (batch, seq_len, 1)
        context = torch.sum(attn_weights_temp * hidden_states, dim=1)
        #weighted_hidden = attn_weights * hidden_states
        #x = self.lstm_layer(weighted_hidden)  

        x = self.dropout(context)
        x = self.fc1(x)
        #return x.squeeze(), attn_weights.squeeze(-1)  # Attention-Gewichte für Analyse
        return x.squeeze(),attention_weights_list_input, attn_weights_temp




class LSTMModelAttentionTemporalTimeShap(nn.Module):
    def __init__(self, input_size, hidden_dim, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_dim)
        self.attention = InputAttention(input_size=input_size, hidden_size=hidden_dim)
        self.lstm_layer = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.temporal_attention = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 1)


    def forward(self, x):
        batch_size, seq_len, num_features = x.size()
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        hidden_states = []
        attention_weights_list_input = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            weighted_x, attention_weights = self.attention(x_t, h_t, c_t)
            h_t, c_t = self.lstm_cell(weighted_x, (h_t, c_t))
            hidden_states.append(h_t.unsqueeze(1))  # (batch, 1, hidden_dim)
            attention_weights_list_input.append(attention_weights)
        hidden_states = torch.cat(hidden_states, dim=1)  # (batch, seq_len, hidden_dim)
        attn_scores = self.temporal_attention(hidden_states)  # (batch, seq_len, 1)
        attn_weights_temp = torch.softmax(attn_scores, dim=1)      # (batch, seq_len, 1)
        context = torch.sum(attn_weights_temp * hidden_states, dim=1)
        #weighted_hidden = attn_weights * hidden_states
        #x = self.lstm_layer(weighted_hidden)  

        x = self.dropout(context)
        x = self.fc1(x)
        #return x.squeeze(), attn_weights.squeeze(-1)  # Attention-Gewichte für Analyse
        return x