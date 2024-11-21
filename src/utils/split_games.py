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

    # Use the base name of the input file for the output prefix
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_prefix = os.path.join(output_dir, f"{base_name}_chunk")

    # Generate the first chunk file name
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

def process_directory(input_dir, output_dir="./data/games", chunk_size_mb=50):
    # Ensure the input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a valid directory.")
        sys.exit(1)

    # Iterate over all .pgn files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".pgn"):
            input_file = os.path.join(input_dir, file_name)
            print(f"Processing file: {input_file}")
            split_games(input_file, output_dir, chunk_size_mb)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    process_directory(input_dir)
