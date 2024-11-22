import chess.pgn
import os
import csv
from concurrent.futures import ProcessPoolExecutor

input_directory = "./data/games/"
output_file = "./data/clean/games_data.csv"

def extract_moves_from_pgn(pgn_file):
    """
    Extracts and returns a list of moves, checkmate status, and game result from a PGN file.
    Args:
        pgn_file (str): Path to the PGN file.
    Returns:
        list: List of rows to be appended to the CSV.
    """
    batch_size = 1000
    outputs = []

    with open(pgn_file, "r", encoding="utf-8") as file:
        print(f"opening file {pgn_file}")
        while True:
            game = chess.pgn.read_game(file)
            if game is None:
                break

            # Extract headers
            white_elo = game.headers.get("WhiteElo")
            black_elo = game.headers.get("BlackElo")
            result = game.headers.get("Result", "Unknown")

            try:
                white_elo = int(white_elo) if white_elo and white_elo.isdigit() else None
                black_elo = int(black_elo) if black_elo and black_elo.isdigit() else None
            except ValueError:
                continue

            # Filter games where either player has an Elo rating over 2200
            if (white_elo and white_elo > 2200) or (black_elo and black_elo > 2200):
                moves = []
                is_checkmate = False
                node = game

                # Traverse the moves
                while node.variations:
                    next_node = node.variation(0)
                    san_move = node.board().san(next_node.move)
                    moves.append(san_move)
                    node = next_node

                # Filter out games with 3 moves or less
                if len(moves) <= 3:
                    continue

                # Add start and end tokens
                moves_string = "<|startofgame|> " + " ".join(moves) + " <|endofgame|>"
                print(moves_string)

                # Determine if the game ended with a checkmate
                if node.board().is_checkmate():
                    is_checkmate = True

                # Prepare the output row
                outputs.append([moves_string, is_checkmate, result])

                # Return results in batches
                if len(outputs) >= batch_size:
                    batch = outputs[:]
                    outputs = []
                    yield batch

        # Return any remaining outputs
        print(f"Processed file {pgn_file}")
        if outputs:
            yield outputs

def process_file(pgn_file):
    """Helper function to extract data from a single PGN file."""
    results = []
    for batch in extract_moves_from_pgn(pgn_file):
        results.extend(batch)
    return results

def main():
    # Collect all .pgn files in the input directory
    pgn_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith(".pgn")]

    # Process files in parallel
    with ProcessPoolExecutor() as executor, open(output_file, 'w', encoding="utf-8", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Moves", "IsCheckmate", "Outcome"])  # Write the CSV header

        for file_results in executor.map(process_file, pgn_files):
            writer.writerows(file_results)  # Append results to CSV

if __name__ == "__main__":
    main()

