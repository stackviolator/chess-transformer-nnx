import csv
import re
import os

# Directory and file paths
input_dir = './data/clean/'
output_file = 'chess_games.csv'

# Regular expressions for parsing games
game_pattern = re.compile(r"<\|startofgame\|>(.*?)<\|endofgame\|>", re.DOTALL)
checkmate_pattern = re.compile(r"(1-0|0-1|1/2-1/2)")

# Gather all input files matching the pattern
input_files = [
    os.path.join(input_dir, filename)
    for filename in os.listdir(input_dir)
    if re.match(r'lichess_chunk_\d+\.txt', filename)
]

# Ensure the input directory contains files to process
if not input_files:
    print("No matching files found in the specified directory.")
    exit()

# Write headers only once
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["moves", "is_checkmate", "outcome"])
    writer.writeheader()

# Process each file and append data to the CSV
for input_file in input_files:
    games_data = []
    with open(input_file, 'r') as file:
        content = file.read()
        games = game_pattern.findall(content)
        for game in games:
            moves = game.strip().splitlines()[:-1]  # Exclude outcome from moves
            outcome_line = game.strip().splitlines()[-1]  # Get the outcome
            outcome = outcome_line if checkmate_pattern.match(outcome_line) else "unknown"
            is_checkmate = outcome in ("1-0", "0-1")  # Checkmate occurs for win/lose outcomes

            games_data.append({
                "moves": " ".join(moves),
                "is_checkmate": is_checkmate,
                "outcome": outcome
            })

    # Append the processed data to the CSV file
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["moves", "is_checkmate", "outcome"])
        writer.writerows(games_data)

    print(f"Processed and appended data from {input_file} to {output_file}.")

print(f"All data has been successfully processed and written to {output_file}.")
