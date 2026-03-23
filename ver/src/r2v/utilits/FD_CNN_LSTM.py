import torch
import torch.nn as nn

class CNN_LSTM_Classifier(nn.Module):
    def __init__(self, in_channels, hidden_dim, kernel_size, num_class):
        super(CNN_LSTM_Classifier, self).__init__()
        self.num_class = num_class
        self.padding = (kernel_size - 1) // 2

        self.cnn1 = nn.Conv1d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=kernel_size, padding=self.padding)
        self.cnn2 = nn.Conv1d(in_channels=hidden_dim, out_channels=64, kernel_size=kernel_size, padding=self.padding)

        self.lstm1 = nn.LSTM(input_size=64, hidden_size=128, bidirectional=True, batch_first=True)

        self.fc = nn.Linear(128 * 2 , self.num_class)

    def forward(self, x):
        x = x.view(1, x.size(0), x.size(1)) 
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, features_dim, seq_len)
        x = self.cnn1(x)
        x = self.cnn2(x) 
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, seq_len, hidden_dim)
        x, _ = self.lstm1(x)

        x = torch.mean(x, dim=1)  
        x = self.fc(x)
        return x  

    