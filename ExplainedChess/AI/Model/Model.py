import torch.nn as nn
from AI.Model.ModelConfig import ModelConfig


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        input_size = ModelConfig.input_row_size * ModelConfig.input_number_of_rows
        hidden_size = int(2 / 3 * input_size) + 1

        self.fc1 = nn.Linear(input_size, hidden_size).cuda()
        self.fc2 = nn.LSTM(hidden_size, hidden_size).cuda()
        self.fc3 = nn.Linear(hidden_size, 1).cuda()

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)

        # Flatten the input tensor for LSTM
        x = x.unsqueeze(0)  # Adding seq_len dimension (assuming seq_len=1 for simplicity)

        # Pass through LSTM (fc3)
        output, (h_n, c_n) = self.fc2(x)

        # You might want to use output or h_n as input to the next layer

        # Example: Pass output through a linear layer followed by sigmoid activation
        x = output.squeeze(0)  # Remove seq_len dimension
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x
