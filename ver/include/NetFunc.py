import torch
import torch.nn as nn
from torch_geometric.nn import GATConv



    

def get_graph(raw_data, sensor_num = 6):
    sensor_data = raw_data

    n = sensor_data.shape[0] # sequence_length
    node_features = sensor_data.contiguous().view(-1, 3) 
    node_features = node_features.float()

    edge_index = []
    edge_type = []

    for t in range(n-1): 
        for sensor_id in range(sensor_num): 
            edge_index.append([sensor_num * t + sensor_id, sensor_num * (t+1) + sensor_id])  
            edge_type.append(0) 

    for t in range(n):
        edge_index.append([sensor_num * t, sensor_num * t + 2])  # Acc -> Gyro
        edge_index.append([sensor_num * t + 4, sensor_num * t + 2])  # Mag -> Gyro
        edge_index.append([sensor_num * t + 2, sensor_num * t])  # Gyro -> Acc
        edge_index.append([sensor_num * t + 2, sensor_num * t + 4])  # Gyro -> Mag
        edge_type.extend([1, 2, 3, 4])  

    for t in range(n):
        edge_index.append([sensor_num * t, sensor_num * t + 1])  # Acc -> Error Acc
        edge_index.append([sensor_num * t + 2, sensor_num * t + 3])  # Gyro -> Error Gyro
        edge_index.append([sensor_num * t + 4, sensor_num * t + 5])  # Mag -> Error Mag
        edge_type.extend([5, 5, 5])  

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long) 

    return node_features, edge_index, edge_type

def get_graph_chain(raw_data, sensor_num=6):
    sensor_data = torch.tensor(raw_data, dtype=torch.float)

    
    n = sensor_data.shape[0] # sequence_length
    node_features = sensor_data.contiguous().view(-1, 3)  

    edge_index = []

    for t in range(n-1):
        for sensor_id in range(sensor_num):  # 0: Acc, 1: Err_Acc 2: Gyro, 3: Err_Gyro 4: Mag 5: Err_Mag
            edge_index.append([sensor_num * t + sensor_id, sensor_num * (t+1) + sensor_id])  
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return node_features, edge_index

def generate_chain_edge_index(num_nodes):
    row = torch.arange(num_nodes - 1)
    col = torch.arange(1, num_nodes)
    
    edge_index = torch.stack([row, col], dim=0)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    return edge_index
