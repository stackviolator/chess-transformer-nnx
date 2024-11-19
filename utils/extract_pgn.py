import chess.pgn
import os
from concurrent.futures import ProcessPoolExecutor

def process_file(file):
    input_file_dir = "./data/games/"
    output_file_dir = "./data/clean/"

    output_data = []
    if file.endswith(".pgn"):
        input_file = os.path.join(input_file_dir, file)
        output_file = os.path.join(output_file_dir, file[:-4] + ".txt")

        with open(input_file, "r", encoding="utf-8") as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                # Extract the headers
                white_elo = game.headers.get("WhiteElo")
                black_elo = game.headers.get("BlackElo")
                result = game.headers.get("Result")

                try:
                    white_elo = int(white_elo) if white_elo and white_elo.isdigit() else None
                    black_elo = int(black_elo) if black_elo and black_elo.isdigit() else None
                except ValueError:
                    continue

                # Check if either player has an Elo rating over 2200
                if (white_elo and white_elo > 2200) or (black_elo and black_elo > 2200):
                    # Extract the move sequence
                    moves = game.mainline_moves()

                    o = ['<|startofgame|>']
                    for m in moves:
                        o.append(str(m))
                    o.append(result)
                    o.append('<|endofgame|>')

                    output_data.append(o)

        # Write the output data
        with open(output_file, 'w') as f:
            print(f"Writing data to {output_file}")
            for game in output_data:
                for move in game:
                    f.write(move)
                    f.write("\n")

    return f"Processed {file}"

def main():
    input_file_dir = "./data/games/"

    # Collect all PGN files
    files = [file for _, _, files in os.walk(input_file_dir) for file in files if file.endswith(".pgn")]

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_file, files)

    for result in results:
        print(result)

if __name__ == "__main__":
    main()

