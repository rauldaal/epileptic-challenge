import torch
from torch import nn

class EpilepsMoedel(nn.Module):
    def __init__(self, **kwargs):
        super(EpilepsMoedel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Conv1d(),
            nn.MaxPool1d(),
            nn.Linear(),
            nn.Dropout(),
            nn.Linear,
            nn.Sigmoid()
        )

    def forward(self, x):
        self.network(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.fc1 = nn.Linear(output_size, 16)    
        self.sig = nn.Sigmoid()


    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = self.fc1(out)
        out = self.sig(out)

        return out
