import torch

from AI.Model.Model import Model
from AI.Model.ModelConfig import ModelConfig
from AI.Utils.Utils import Utils
import torch.nn.functional as F


class Explainer:
    def __init__(self):
        self.model = Model()
        self.model.load_state_dict(torch.load("./AI/model.pth"))
        self.model.eval()

        self.words, self.word_value_table = self.__load_word_value_table()

    def explain(self, question):
        answer = []
        lng, question_tensor = self.get_question_tensor(question)

        output = self.model(question_tensor.unsqueeze(1))
        word_output = self.get_word_from_value(output.item())

        i = 1
        while word_output != '.' and i < ModelConfig.input_layer_size:
            answer.append(word_output)
            question_tensor = torch.cat([question_tensor[0:lng], output], dim=0)
            lng += 1
            question_tensor = F.pad(question_tensor, pad=(0, ModelConfig.input_layer_size - lng))
            output = self.model(question_tensor.unsqueeze(1))
            word_output = self.get_word_from_value(output.item())

            i += 1
            
        answer_string = ""
        for word in answer:
            answer_string += word + " "

        return answer_string

    @staticmethod
    def __load_word_value_table():
        word_value_table = {}
        words = []
        with open("./AI/word_to_index.txt") as file:
            line = file.readline()
            while line:
                word, index = line.split()
                words.append(word)
                word_value_table[word] = float(index)
                line = file.readline()

        return words, word_value_table

    def get_question_tensor(self, question):
        words = Utils.preprocess_question(question)
        word_values = [self.word_value_table[word] for word in words]
        padded_tensor = F.pad(torch.tensor(word_values), pad=(0, ModelConfig.input_layer_size - len(word_values)))
        return len(word_values), padded_tensor.float()

    def get_word_from_value(self, value):
        for i in range(0, len(self.words) - 1):
            if self.word_value_table[self.words[i]] <= value < self.word_value_table[self.words[i + 1]]:
                return self.words[i]
        return self.words[-1]
