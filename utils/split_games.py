import os
import sys

def split_games(input_file, output_dir="./data/games", chunk_size_mb=50):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define chunk size in bytes
    chunk_size = chunk_size_mb * 1024 * 1024

    # Initialize variables
    current_size = 0
    chunk_index = 1
    output_prefix = os.path.join(output_dir, "lichess_chunk")
    output_file = f"{output_prefix}_{chunk_index}.pgn"

    # Open the first output file
    with open(output_file, 'w') as out_file:
        print(f"Created chunk file: {output_file}")

        # Read the input file line by line
        with open(input_file, 'r') as in_file:
            for line in in_file:
                # Calculate the size of the line in bytes
                line_size = len(line.encode('utf-8'))

                # Check if adding this line exceeds the chunk size
                if current_size + line_size > chunk_size:
                    # If the line is a game separator (blank line), start a new chunk
                    if line.strip() == "":
                        # Finalize the current chunk
                        out_file.write("\n")

                        # Move to the next chunk
                        chunk_index += 1
                        output_file = f"{output_prefix}_{chunk_index}.pgn"
                        out_file = open(output_file, 'w')
                        print(f"Created chunk file: {output_file}")
                        current_size = 0
                        continue

                # Write the line to the current output file
                out_file.write(line)

                # Update the current size
                current_size += line_size

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    split_games(input_file)
