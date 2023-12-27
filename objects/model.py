import torch
from torch import nn

class EpilepsMoedel(nn.Module):
    def __init__(self, **kwargs):
        super(EpilepsMoedel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Conv1d(in_channels=21, out_channels=5, kernel_size=6),
            nn.MaxPool1d(kernel_size=4, stride=1, padding=1),
            nn.Flatten(),  # Necesario para aplanar la salida antes de la capa lineal
            nn.Linear(5 * 128, 2),  # Ajusta la entrada de acuerdo con la salida de la capa anterior
            nn.Dropout(p=0.2),
            nn.Linear(2, 1),
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
