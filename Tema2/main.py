import chess.pgn

counter = dict()
for i in range(1, 1000):
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
