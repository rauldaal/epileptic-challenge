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
            nn.Dropout(p=0.15),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.network(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self, **kwargs):
        super(LSTMModel, self).__init__()
        self.hidden_size = 64
        self.input_size = 128
        self.num_layers = 8
        self.batch_size = 64
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size, 1)
        self.sig = nn.Sigmoid()

        self.hidden_state = self.init_hidden(self.hidden_size, self.num_layers, self.batch_size)


    def init_hidden(self, hidden_size, num_layers, batch_size):
        # Inicializar el estado oculto y de celda con tensores de ceros
        return (torch.zeros(num_layers, batch_size, hidden_size),
                torch.zeros(num_layers, batch_size, hidden_size))

    def forward(self, x):
        if x.shape[1] != 128:
            x = x.permute(0, 2, 1)
        out, _ = self.lstm(x, self.hidden_state)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.sig(out)

        return out
