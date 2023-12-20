from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

CORPUS_FILE = "chat.txt"

chatbot = ChatBot("Chatpot")

trainer = ListTrainer(chatbot)
with open(CORPUS_FILE, "r") as corpus:
    cleaned_corpus = [line.strip() for line in corpus.readlines()]

trainer.train(cleaned_corpus)

exit_conditions = (":q", "quit", "exit")
while True:
    query = input("> ")
    if query in exit_conditions:
        break
    else:
        answer = chatbot.get_response(query)
        while '?' in str(answer):
            answer = chatbot.get_response(query)
        print(answer)
