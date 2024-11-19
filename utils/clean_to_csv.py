import csv
import re

# File paths
input_file = 'data/clean/lichess_chunk_16.txt'  # Update this path if needed
output_file = 'chess_games.csv'

# Regular expressions for parsing games
game_pattern = re.compile(r"<\|startofgame\|>(.*?)<\|endofgame\|>", re.DOTALL)
checkmate_pattern = re.compile(r"(1-0|0-1|1/2-1/2)")

# Process the file and extract data
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

# Write to CSV
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["moves", "is_checkmate", "outcome"])
    writer.writeheader()
    writer.writerows(games_data)

print(f"Data has been successfully written to {output_file}.")
