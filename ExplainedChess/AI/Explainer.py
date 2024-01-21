import torch
from stockfish import Stockfish

from AI.Model.Model import Model
from AI.Utils.Utils import Utils

import chess


class Explainer:
    def __init__(self, dataset_path):

        self.data = Utils.load_dataset(dataset_path)
        self.dataset_length = len(self.data)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = Model().to(self.device)
        self.model.load_state_dict(torch.load("./AI/model.pth"))
        self.model.eval()

    def explain(self, question):
        question_vector = Utils.process_question(question)
        question_vector = torch.tensor(question_vector, dtype=torch.float32)
        question_vector = question_vector.view(1, -1)
        question_vector = question_vector.to(self.device)

        output = self.model(question_vector)
        output = int(output.item() * self.dataset_length)

        return self.data[output]['answer']

    def get_next_set_of_moves_from_moves(self, question):
        question = Utils.preprocess_input(question)
        question = Utils.remove_punctuation(question)

        tokens = question.split(' ')

        moves = []
        first_move_found = False
        end_of_moves = False
        number_of_moves_ahead = None
        for token in tokens:
            if end_of_moves is False and self.is_token_first_move(token):
                moves.append(token)
                first_move_found = True
            elif end_of_moves is False and first_move_found:
                if self.is_token_move(token):
                    moves.append(token)
                else:
                    end_of_moves = True
            else:
                if end_of_moves is True and '1' <= token <= '9':
                    if number_of_moves_ahead is None:
                        number_of_moves_ahead = int(token)

        board = chess.Board()
        for move in moves:
            board.push_san(move)

        fen = board.fen()

        stockfish = Stockfish("./AI/stockfish")
        stockfish.set_fen_position(fen)

        answer = "Pozitia curenta este: " + fen + "\n\n"
        answer = "Cele mai bune mutari sunt: "
        best_player_to_move_first = ""
        for i in range(number_of_moves_ahead):
            best_move = stockfish.get_best_move()
            answer += best_move + " "
            if i % 2 == 0:
                best_player_to_move_first += best_move + " "
            stockfish.make_moves_from_current_position([best_move])

        answer += "\n\n"
        answer += "Deci cele mai bune mutari ale jucatorului curent sunt: " + best_player_to_move_first
        answer += "\n"
        answer += "Poziția după aceste mutări este: " + stockfish.get_fen_position()

        probability_to_win = stockfish.get_evaluation()['value']

        if probability_to_win > 0:
            answer += "\n\n"
            answer += "Jucatorul curent are o probabilitate de castig de " + str(probability_to_win) + "%"
        elif probability_to_win < 0:
            answer += "\n\n"
            answer += "Jucatorul curent are o probabilitate de pierdere de " + str(probability_to_win) + "%"
        else:
            answer += "\n\n"
            answer += "Jucatorul curent are o probabilitate de remiza de " + str(probability_to_win) + "%"

        return answer


    @staticmethod
    def is_token_first_move(token):
        if token[1] == '3' or token[1] == '4':
            if 'h' >= token[0] >= 'a':
                return True
        return False

    @staticmethod
    def is_token_move(token):
        if len(token) > 5:
            return False
        for char in token:
            if not (
                    'h' >= char >= 'a' or '8' >= char >= '1' or char == 'x' or char == '-' or char == '+' or char == '#'):
                return False
        return True
