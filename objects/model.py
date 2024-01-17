import torch
from torch import nn
import torch.nn.functional as F

from objects import(ModelWeightsInit)

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
class EpilepsyLSTM(nn.Module):
    """
    Implementation:
        A channel independent generalized seizure detection method for pediatric epileptic seizures
        batch_size 600
        epochs 1000
        lr = 1e-4
        optmizer Adam
    """
    def __init__(self, inputmodule_params,net_params,outmodule_params):
        super().__init__()

        print('Running class: ', self.__class__.__name__)
        
        ### NETWORK PARAMETERS
        n_nodes=inputmodule_params['n_nodes']
    
        Lstacks=net_params['Lstacks']
        dropout=net_params['dropout'] 
        hidden_size=net_params['hidden_size']
       
        # n_classes=outmodule_params['n_classes']
        n_classes=1
        hd=outmodule_params['hd']
        
        self.inputmodule_params=inputmodule_params
        self.net_params=net_params
        self.outmodule_params=outmodule_params
        
        ### NETWORK ARCHITECTURE
        # IF batch_first THEN (batch, timesteps, features), ELSE (timesteps, batch, features)
        self.lstm = nn.LSTM(input_size=n_nodes, # the number of expected features (out of convs)
                                       hidden_size= hidden_size, # the number of features in the hidden state h
                                       num_layers= Lstacks, # number of stacked lstms 
                                       batch_first = True,
                                       bidirectional = False,
                                       dropout=dropout)

        self.fc = nn.Sequential(nn.Linear(hidden_size, hd),
                                nn.ReLU(),
                                nn.Linear(hd, n_classes),
                                nn.Sigmoid()
                                ) 

    def init_weights(self):
         ModelWeightsInit.init_weights_xavier_normal(self)
        
    def forward(self, x):
        
        ## Reshape input
        # input [batch, features (=n_nodes), sequence_length (T)] ([N, 21, 640])
        x = x.permute(0, 2, 1) # lstm  [batch, sequence_length, features]
        
        ## LSTM Processing
        out, (hn, cn) = self.lstm(x)
        # out is [batch, sequence_length, hidden_size] for last stack output
        # hn and cn are [1, batch, hidden_size]
        out = out[:, -1, :] # hT state of lenght hidden_size

        ## Output Classification (Class Probabilities)
        x = self.fc(out)

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


