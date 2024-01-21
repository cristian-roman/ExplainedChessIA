from math import log2
import nltk
from AI.Model.ModelConfig import ModelConfig
import json


class Utils:

    @staticmethod
    def create_json_file_from_plain_text(file, output_file):
        with open(file) as f:
            question = f.readline()
            i = 0
            data = []
            while question:
                answer = f.readline()
                question = Utils.preprocess_input(question)
                answer = Utils.preprocess_input(answer)
                data_row = {'order_number': i, 'question': question, 'answer': answer}
                data.append(data_row)
                question = f.readline()
                i += 1
        with open(output_file, 'w') as outfile:
            json.dump(data, outfile)

    @staticmethod
    def preprocess_input(x, to_lower=True):
        if to_lower:
            x = x.lower()
        x = x.replace('\n', '')
        x = x.replace('.', ' .')
        x = x.replace(',', ' ,')
        x = x.replace('?', ' ?')
        x = x.replace('!', ' !')
        x = x.replace(';', ' ;')
        x = x.replace(':', ' :')
        x = x.replace('(', ' ( ')
        x = x.replace(')', ' ) ')
        x = x.replace('[', ' [ ')
        x = x.replace(']', ' ] ')
        x = x.replace('{', ' { ')
        x = x.replace('}', ' } ')
        x = x.replace('  ', ' ')
        return x

    @staticmethod
    def load_dataset(dataset_path):
        with open(dataset_path, 'r') as json_file:
            data = json.load(json_file)
            return data

    @staticmethod
    def process_question(question):
        question = Utils.remove_punctuation(question)

        # tokenize
        tokens = nltk.word_tokenize(question)

        # remove stop words
        stop_words = set(nltk.corpus.stopwords.words('romanian'))
        tokens = [w for w in tokens if not w in stop_words]

        # stemming
        stemmer = nltk.stem.PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]

        question_vector = []
        for token in tokens:
            i = 0
            for letter in token:
                question_vector.append(float(ord(letter)))
                i += 1
            while i < ModelConfig.input_row_size:
                question_vector.append(0)
                i += 1

        number_of_rows = len(tokens)

        while number_of_rows < ModelConfig.input_number_of_rows:
            i = 0
            while i < ModelConfig.input_row_size:
                question_vector.append(0)
                i += 1
            number_of_rows += 1

        return question_vector

    @staticmethod
    def remove_punctuation(question):
        question = question.replace('.', '')
        question = question.replace(',', '')
        question = question.replace('?', '')
        question = question.replace('!', '')
        question = question.replace(';', '')
        question = question.replace(':', '')
        question = question.replace('(', '')
        question = question.replace(')', '')
        question = question.replace('[', '')
        question = question.replace(']', '')
        question = question.replace('{', '')
        question = question.replace('}', '')
        question = question.replace("\"", '')
        return question
