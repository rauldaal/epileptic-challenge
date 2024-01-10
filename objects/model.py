import torch
from torch import nn

class EpilepsModel(nn.Module):
    def __init__(self, **kwargs):
        super(EpilepsModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Conv1d(in_channels=21, out_channels=5, kernel_size=6),
            nn.MaxPool1d(kernel_size=4, stride=1, padding=1),
            nn.Flatten(),  # Necesario para aplanar la salida antes de la capa lineal
            nn.Linear(610, 2),  # Ajusta la entrada de acuerdo con la salida de la capa anterior
            nn.Dropout(p=0.2),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.network(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self, **kwargs):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=21, hidden_size=16, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(16, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.sig(out)

        return out
