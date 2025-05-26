import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Maxout(nn.Module):
    def __init__(self, input_features, output_features, num_units):
        super(Maxout, self).__init__()
        self.num_units = num_units
        self.linear = nn.Linear(input_features, output_features * num_units)

    def forward(self, x):
        shape = x.shape
        out = self.linear(x)
        out = out.view(shape[0], -1, self.num_units)
        out, _ = torch.max(out, dim=2)
        return out

class Model_LSTM(nn.Module):
    def __init__(self):
        super(Model_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=2560, hidden_size=512, num_layers=1, batch_first=True)
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 1)
        self.activation1 = Maxout(256, 64, num_units=4)
        self.activation2 = Maxout(32, 32, num_units=1)
        self.dropout1 = nn.Dropout(0.2)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:,-1,:]
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        return x
    