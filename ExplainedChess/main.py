from AI.Trainer import Trainer
from Server.server import Server
trainer = Trainer("./AI/Data/chat.txt")
trainer.train()

server = Server()
server.run()

