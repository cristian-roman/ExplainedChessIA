import torch
import torch.nn as nn
from torch.nn import init


class Model(nn.Module):
    input_size = 750
    hidden_size = 750
    num_layers = 2
    output_size = 1

    def __init__(self):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(Model.input_size, Model.hidden_size, Model.num_layers)
        init.xavier_normal_(self.lstm.weight_ih_l0)
        self.linear = nn.Linear(Model.hidden_size, Model.output_size)
        init.xavier_normal_(self.linear.weight)

    def forward(self, input):
        output, _ = self.lstm(input)
        output = self.linear(output[:, -1])
        output = torch.sigmoid(output)
        return output
