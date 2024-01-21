from AI.Trainer import Trainer
from AI.Utils.Utils import Utils
from Server.server import Server

#Utils.create_json_file_from_plain_text("./AI/Data/chat.txt", "./AI/Data/chat.json")
#trainer = Trainer("./AI/Data/chat.json")
#trainer.train()

server = Server()
server.run()

