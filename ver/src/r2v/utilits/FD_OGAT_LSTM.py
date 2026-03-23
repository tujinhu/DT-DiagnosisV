import torch
import torch.nn as nn
from torch_geometric.nn import GATConv



class OGAT_LSTM_Classifier(nn.Module):
    def __init__(self, in_channels, hidden_dim, gru_hidden_dim, num_heads, sensor_num, num_class):
        super(OGAT_LSTM_Classifier, self).__init__()
        self.sensor_num = sensor_num
        self.num_class = num_class

        self.gat1 = GATConv(in_channels, hidden_dim, heads=num_heads)
        self.lstm1 = nn.LSTM(hidden_dim * num_heads * self.sensor_num, gru_hidden_dim, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(gru_hidden_dim, self.num_class)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index) 
        x = torch.sigmoid(x) * x
        x = x.view(1, -1, x.size(-1) * self.sensor_num) 
        x, _ = self.lstm1(x)  
        x = torch.mean(x, dim=1)  
        x = self.fc(x)
        return x 