import chess.pgn

# with open('games.txt', 'r') as file:
#     games_text = file.read()
#
# games_text.replace('\r\n', ' ')
#
# partide = games_text.split('#')
# nr_partida = 0
# for partida in partide:
#     nr_partida += 1
#     mutari = partida.split()
#     if len(mutari) == 0:
#         continue
#     with open(f'partida_{nr_partida}.pgn', 'w') as file:
#         spatiu = 0
#         nr_mutari = len(mutari)
#         for i in range(0, nr_mutari - 1):
#             mutare = mutari[i]
#             spatiu += 1
#             if spatiu % 2 == 1:
#                 file.write(str(spatiu//2 + 1) + '. ' + mutare + ' ')
#             if spatiu % 2 == 0:
#                 file.write(mutare + ' ')
#
#         if spatiu % 2 == 0:
#             file.write(str(spatiu//2 + 1) + '. ' + mutari[-1] + '# ')
#             file.write('*')
#
#         else:
#             file.write(mutari[-1] + '# ')


counter = dict()

for i in range(1, 6326):
    file_path = f"partida_{i}.pgn"

    with open(file_path, "r") as pgn_file:
        game = chess.pgn.read_game(pgn_file)

        if game is not None:
            last_6_moves = []

            for move in game.mainline_moves():
                last_6_moves.append(move)

                if len(last_6_moves) > 6:
                    last_6_moves.pop(0)

            if len(last_6_moves) == 6:
                sep = 0
                first_player_moves = []
                second_player_moves = []

                for move in last_6_moves:
                    sep += 1

                    if sep % 2 == 1:
                        first_player_moves.append(move)
                    else:
                        second_player_moves.append(move)

                first_player_moves_as_string = " ".join(str(move) for move in first_player_moves)
                second_player_moves_as_string = " ".join(str(move) for move in second_player_moves)

                counter[first_player_moves_as_string] = counter.get(first_player_moves_as_string, 0) + 1
                counter[second_player_moves_as_string] = counter.get(second_player_moves_as_string, 0) + 1

for key in sorted(counter, key=counter.get, reverse=True):
    value = counter[key]

    if value > 2:
        print(key, value)