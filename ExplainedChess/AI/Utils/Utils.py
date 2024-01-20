import nltk


class Utils:
    @staticmethod
    def download_resources():
        nltk.download('stopwords')
        nltk.download('punkt')

    @staticmethod
    def preprocess_question(question):
        words = Utils.preprocess_sentence(question)
        words = Utils.remove_stopwords(words)
        return words

    @staticmethod
    def preprocess_sentence(sentence):
        words = Utils.tokenize(sentence)
        words = Utils.remove_punctuation(words)
        return words

    @staticmethod
    def tokenize(sentence):
        return nltk.word_tokenize(sentence)
    @staticmethod
    def remove_stopwords(words):
        stopwords = nltk.corpus.stopwords.words('romanian')
        return [word for word in words if word not in stopwords]

    @staticmethod
    def remove_punctuation(words):
        return [word for word in words if word.isalpha()]
