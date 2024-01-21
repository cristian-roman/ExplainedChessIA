import torch
import torch.nn as nn
from torch.nn import init

from AI.Model.ModelConfig import ModelConfig


class Model(nn.Module):
    num_layers = 2
    output_size = 1

    def __init__(self):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(ModelConfig.input_layer_size, ModelConfig.input_layer_size, Model.num_layers)
        init.xavier_normal_(self.lstm.weight_ih_l0)
        self.linear = nn.Linear(ModelConfig.input_layer_size, Model.output_size)
        init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.linear(output[:, -1])
        output = torch.sigmoid(output)
        return output
