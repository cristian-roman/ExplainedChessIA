import torch
from torch import nn

from AI.Model.Model import Model
from AI.Utils.Utils import Utils


class Trainer:
    def __init__(self, dataset_path, minimal_loss_path="./AI/loss.txt"):
        self.data = Utils.load_dataset(dataset_path)
        self.dataset_length = len(self.data)

        self.minimal_loss = 100000
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("Do you want to load the model from a file? (y/n)")
        answer = input()
        if answer == 'y':
            self.model = Model().to(self.device)
            self.model.load_state_dict(torch.load("./AI/model.pth"))

            with open(minimal_loss_path, 'r') as f:
                self.minimal_loss = float(f.readline())
        else:
            self.model = Model().to(self.device)  # Move the model to GPU
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.000001)

    def train(self):
        total_loss = 0.0
        for epoch in range(1, 10000):
            current_loss = 0.0
            for i in range(self.dataset_length):
                question = self.data[i]['question']
                question_vector = Utils.process_question(question)
                question_vector = torch.tensor(question_vector, dtype=torch.float32).to(
                    self.device)  # Move input to GPU
                question_vector = question_vector.view(1, -1)

                output = self.model(question_vector)
                expected_output = float(self.data[i]['order_number']) / self.dataset_length
                expected_output = torch.tensor(expected_output, dtype=torch.float32).view(1, 1).to(self.device)

                loss = self.criterion(output, expected_output)
                current_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += current_loss
            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, average_loss: {total_loss / 10}\n')
                total_loss = 0.0

            if current_loss < self.minimal_loss:
                self.minimal_loss = current_loss
                torch.save(self.model.state_dict(), "./AI/model.pth")
                print(f"best model saved with loss: {self.minimal_loss} at epoch: {epoch}")
                with open("./AI/loss.txt", 'w') as f:
                    f.write(str(self.minimal_loss))
