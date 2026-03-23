import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class OGCN_GRU_Classifier(nn.Module):
    def __init__(self, in_channels, hidden_dim, gru_hidden_dim, sensor_num, num_class):
        super(OGCN_GRU_Classifier, self).__init__()
        self.sensor_num = sensor_num
        self.num_class = num_class

        self.gcn1 = GCNConv(in_channels, hidden_dim)
        self.gru1 = nn.GRU(hidden_dim * self.sensor_num, gru_hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(gru_hidden_dim * 2, self.num_class)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index) 
        x = torch.sigmoid(x) * x
        x = x.view(1, -1, x.size(-1) * self.sensor_num) 
        x, _ = self.gru1(x)  
        x = torch.mean(x, dim=1)  
        x = self.fc(x)
        return x 