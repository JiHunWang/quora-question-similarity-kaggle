
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes, bidirectional=False):
        super(SiameseNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_classes = num_classes
        
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)
        self.fc_1 = nn.Linear(self.hidden_size, 128)
        self.fc_2 = nn.Linear(128, 32)
        self.fc_3 = nn.Linear(32, self.num_classes)
    
    def forward_common(self, x):
    	x = torch.unsqueeze(x, dim=1)
        h_0, c_0 = None, None
        if self.bidirectional:
            h_0 = Variable(torch.zeros(self.num_layers, x.size[0], self.hidden_size))
            c_0 = Variable(torch.zeros(self.num_layers, x.size[0], self.hidden_size))
        else:
            h_0 = Variable(torch.zeros(2 * self.num_layers, x.size[0], self.hidden_size))
            c_0 = Variable(torch.zeros(2 * self.num_layers, x.size[0], self.hidden_size))
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        output = self.relu(output)
        output = self.fc_1(output.view(-1, self.hidden_size))
        output = self.relu(output)
        output = self.fc_2(output)
        output = self.relu(output)
        output = self.fc_3(output)
        return output
    
    def forward(self, x_1, x_2):
        output1 = forward_common(x_1)
        output2 = forward_common(x_2)
        return output1, output2
