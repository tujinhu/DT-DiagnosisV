import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


Epsilon_ij = {
        'a2g': 0.9646226410483146,
        'm2g': 0.9994889459516024,
        'g2a': 0.9646226410483146,
        'g2m': 0.9994889459516024
    }

# Epsilon_ij = {
#         'a2g': 0.8362533674285337,
#         'm2g': 0.9958066048617867,
#         'g2a': 0.8362533674285337,
#         'g2m': 0.9958066048617867
#     }

class MyGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0.3, gamma=1.0, **kwargs):
        super(MyGATConv, self).__init__(aggr='add', **kwargs)
        '''
        in_channels, out_channels: dimensions of input and output features.
        heads: number of heads of the attention mechanism, indicating how many independent attention mechanisms are used in the multi-head attention mechanism.
        concat: whether to concatenate the outputs of multiple heads together.
        negative_slope: negative slope parameter of the LeakyReLU activation function.
        dropout: used to randomly drop part of the attention weights to prevent overfitting.
        gamma: weight of the penalty term, used to control the effect of the distance penalty on the attention weights.
        self.weight: stores the weight of the linear transformation from input features to output features.
        self.att: stores the parameters used to calculate the attention weights.
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.gamma = gamma

        self.weight = nn.Parameter(torch.Tensor(heads, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(heads, 2 * out_channels))

        self.leaky_relu = nn.LeakyReLU(self.negative_slope)
        self.attention_dropout = nn.Dropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, edge_type):
        # Step 1: Linearly transform node feature matrix.
        '''
        Shape:
        x: [N, in_channels]  (Physical Meaning: Each node(N) has in_channels's features (3-axis sensoor data))
        x.unsqueeze(1): [N, 1, in_channels]
        self.weight: [num_heads, in_channels, out_channels]
        x('nid,hdo->nho'): [N, num_heads, out_channels] (Physical Meaning: Each head(num_heads) of each node(N) outputs features(out_channels))
        '''
        x = torch.einsum('nid,hdo->nho', x.unsqueeze(1), self.weight)

        # Step 2: Start propagating messages.
        # 2.1）: Compute attention coefficients using custom logic.
        '''
        Shape:
        x_i: [E, num_heads, out_channels] (Source node features)
        x_j: [E, num_heads, out_channels] (Target node features)
        x_concat: [E, num_heads, 2 * out_channels]
        self.att: [E, 2 * out_channels]
        alpha: [E, num_heads]
        distance: [E, num_heads]

        att^T [Wh_i || Wh_j] --> [E, num_heads, 2 * out_channels] --> sum(dim=-1) --> [E, num_heads]
        '''
        row, col = edge_index
        x_i = x[row]
        x_j = x[col]
        x_concat = torch.cat([x_i, x_j], dim=-1)
        alpha = (x_concat * self.att).sum(dim=-1) # alpha = torch.einsum('eno,no->eno', x_concat, self.att) Dot product instead of matrix multiplication
        alpha = F.leaky_relu(alpha, negative_slope=self.negative_slope)

        # # 2.1.1）: Compute the distance between nodes (using L2 distance as an example)
        # distance = torch.norm(x_i - x_j, p=2, dim=-1)
        # # 2.1.2）: Apply the distance-based penalty to the attention scores
        # alpha = alpha + self.gamma * distance

        # 2.1.3）: Assigning weights to different types of edges
        alpha_weights = torch.ones_like(alpha)  # The default weight is 1
        alpha_weights[edge_type == 0] = 1     # Time-dependent edge weight 0.1
        alpha_weights[(edge_type == 1) |(edge_type == 2) | (edge_type == 3) | (edge_type == 4)] = 1     # Coupling edge weight 0.2
        alpha_weights[edge_type == 5] = 1     # Error correction edge weight 0.7

        alpha_weights_Epsilon = torch.ones_like(alpha)     
        alpha_weights_Epsilon[edge_type == 1] = Epsilon_ij['a2g']     
        alpha_weights_Epsilon[edge_type == 2] = Epsilon_ij['m2g']
        alpha_weights_Epsilon[edge_type == 3] = Epsilon_ij['g2a']     
        alpha_weights_Epsilon[edge_type == 4] = Epsilon_ij['g2m']
           
        # 2.1.4）: Apply edge weights
        alpha = alpha * alpha_weights + alpha_weights_Epsilon

        alpha = softmax(alpha, row)
        alpha = self.attention_dropout(alpha)

        # 2.2）: Multiply attention coefficients with neighbor features
        '''
        Shape:
        x_j: [E, num_heads, out_channels] (Target node features)
        alpha: [E, num_heads]
        alpha.unsqueeze(-1): [E, num_heads, 1]
        message: [E, num_heads, out_channels]
        '''
        message = x_j * alpha.unsqueeze(-1)

        # 2.3）: Aggregate the messages from neighboring nodes
        '''
        Shape:
        aggr_out: [N, num_heads * out_channels] (concat = True)
        aggr_out: [N, out_channels] (concat = False)
        '''
        aggr_out = torch.zeros_like(x) # Initialize aggregation output tensor [N, num_heads, out_channels]
        aggr_out.index_add_(0, row, message) # Accumulate messages for each node

        # Handle output shape based on `concat`
        if self.concat:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)  # [N, num_heads * out_channels]
        else:
            aggr_out = aggr_out.mean(dim=1) # [N, out_channels]

        # Return the aggregated features as output
        return aggr_out

class MGAT_GRU_Classifier(nn.Module):
    def __init__(self, in_channels, hidden_dim, gru_hidden_dim, num_heads, sensor_num, num_class):
        super(MGAT_GRU_Classifier, self).__init__()
        self.sensor_num = sensor_num
        self.num_class = num_class

        self.gat1 = MyGATConv(in_channels, hidden_dim, heads=num_heads)
        self.gru1 = nn.GRU(hidden_dim * num_heads * self.sensor_num, gru_hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(gru_hidden_dim * 2, self.num_class)

    def forward(self, x, edge_index, edge_type):
        x = self.gat1(x, edge_index, edge_type) 
        x = torch.sigmoid(x) * x
        x = x.view(1, -1, x.size(-1) * self.sensor_num) 
        x, _ = self.gru1(x)  
        x = torch.mean(x, dim=1)  
        x = self.fc(x)
        return x 