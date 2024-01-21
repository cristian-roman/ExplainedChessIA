import random
import torch
from torch import nn, optim
import torch.nn.functional as F
from AI.Model.Model import Model
from AI.Model.ModelConfig import ModelConfig
from AI.Utils.Utils import Utils


class Trainer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.word_value_table, self.shuffled_words = self.__create_vocabulary()

    def train(self):
        ModelConfig.input_layer_size = len(self.shuffled_words)
        lstm_model = Model()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

        for epoch in range(100):
            with open(self.dataset_path) as file:
                question = file.readline()
                while question:
                    lng, question_tensor = self.get_question_tensor(question)

                    answer = file.readline()
                    tokenized_answer = Utils.preprocess_sentence(answer)
                    tokenized_answer.append('.')

                    for i in range(0, len(tokenized_answer)):
                        output = lstm_model(question_tensor.unsqueeze(1))
                        expected_output = torch.tensor([self.word_value_table[tokenized_answer[i]]]).view(-1, 1).float()

                        loss = criterion(output, expected_output[0])
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        question_tensor = torch.cat([question_tensor[0:lng], expected_output[0]], dim=0)
                        lng += 1
                        question_tensor = F.pad(question_tensor, pad=(0, ModelConfig.input_layer_size - lng))

                    question = file.readline()

            if epoch % 10 == 0:
                print('Epoch: {}/{}............. saving'.format(epoch, 100), end=' ')
                torch.save(lstm_model.state_dict(), './AI/model.pth')

    def __create_vocabulary(self):
        word_table = Trainer.get_words_file(self.dataset_path)
        word_table.add('.')

        shuffled_words = Trainer.__shuffle_words(list(word_table))

        word_value_table = Trainer.__create_word_value_table(shuffled_words)
        Trainer.__save_word_value_table(word_value_table)

        return word_value_table, shuffled_words

    @staticmethod
    def get_words_file(dataset_path):
        word_table = set()

        with open(dataset_path) as file:
            for line in file:
                words = Utils.preprocess_sentence(line)
                for word in words:
                    word_table.add(word)

        return word_table

    @staticmethod
    def __shuffle_words(words):
        shuffled_words = list()
        picked_indexes = set()
        for word in words:
            random_index = random.randint(0, len(shuffled_words))
            while random_index in picked_indexes:
                random_index = random.randint(0, len(shuffled_words))
            shuffled_words.insert(random_index, word)
            picked_indexes.add(random_index)

        return shuffled_words

    @staticmethod
    def __create_word_value_table(words):
        word_to_index = dict()
        step = 1 / (len(words) + 1)
        for i in range(1, len(words)+1):
            word_to_index[words[i-1]] = i * step

        return word_to_index

    @staticmethod
    def __save_word_value_table(word_to_index):
        with open('./AI/word_to_index.txt', 'w') as f:
            for word in word_to_index:
                f.write(word + ' ' + str(word_to_index[word]) + '\n')

    def get_question_tensor(self, question):
        words = Utils.preprocess_question(question)
        word_values = [self.word_value_table[word] for word in words]
        padded_tensor = F.pad(torch.tensor(word_values), pad=(0, ModelConfig.input_layer_size - len(word_values)))
        return len(word_values), padded_tensor.float()
