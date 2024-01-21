import torch
from torch import nn

from AI.Model.Model import Model
from AI.Utils.Utils import Utils


class Trainer:
    def __init__(self, dataset_path):
        self.data = Utils.load_dataset(dataset_path)
        self.dataset_length = len(self.data)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("Do you want to load the model from a file? (y/n)")
        answer = input()
        if answer == 'y':
            self.model = Model().to(self.device)
            self.model.load_state_dict(torch.load("./AI/model.pth"))
        else:
            self.model = Model().to(self.device)  # Move the model to GPU
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

    def train(self):
        minimal_loss = 0.006283239782403527
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
                print(f'Epoch: {epoch}, average_loss: {total_loss/10}\n')
                total_loss = 0.0

            if current_loss < minimal_loss:
                minimal_loss = current_loss
                torch.save(self.model.state_dict(), "./AI/model.pth")
                print(f"best model saved with loss: {minimal_loss} at epoch: {epoch}")
                with open("./AI/loss.txt", 'w') as f:
                    f.write(str(minimal_loss))
