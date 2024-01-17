import chess.pgn
#
# with open('games.txt', 'r') as file:
#     games_text = file.read()
#
# games_text.replace('\r\n', ' ')

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
#         file.write(mutari[-1] + '# ')
#         if spatiu % 2 == 1:
#             file.write('*')


counter = dict()
for i in range(1, 6326):
    pgn = open(f"partida_{i}.pgn")

    game = chess.pgn.read_game(pgn)

    last_3_moves = []
    for move in game.mainline_moves():
        last_3_moves.append(move)
        if len(last_3_moves) > 3:
            last_3_moves.pop(0)

        if len(last_3_moves) == 3:
            key = tuple(last_3_moves)
            counter[key] = counter.get(key) + 1

for key in sorted(counter, key=counter.get, reverse=True):
    value = counter[key]
    if value > 2:
        print(key, value)
