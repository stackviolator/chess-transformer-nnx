import chess.pgn
from chess.pgn import read_game
import os
import csv

input_file = "./data/raw/lichess_db_standard_rated_2024-10.pgn"
output_file_dir = "./data/clean/"

def extract_moves_from_pgn(pgn_file):
    """
    Extracts and saves a list of moves, checkmate status, and game result from a PGN file into a CSV.
    Args:
        pgn_file (str): Path to the PGN file.
    """
    output_file = output_file_dir + 'games_data.csv'
    batch_size = 1000
    outputs = []

    with open(pgn_file, "r", encoding="utf-8") as file, open(output_file, 'w', encoding="utf-8", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Moves", "IsCheckmate", "Outcome"])  # CSV header

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

                # Determine if the game ended with a checkmate
                if node.board().is_checkmate():
                    is_checkmate = True

                # Prepare the output row
                outputs.append([moves_string, is_checkmate, result])

                # Write to CSV in batches
                if len(outputs) >= batch_size:
                    print(f"Writing {len(outputs)} rows to {output_file}")
                    writer.writerows(outputs)
                    outputs = []  # Clear the buffer

        # Final write for any remaining outputs
        if outputs:
            print(f"Writing {len(outputs)} rows to {output_file}")
            writer.writerows(outputs)

def main():
    # Collect all PGN files
    extract_moves_from_pgn(input_file)

if __name__ == "__main__":
    main()