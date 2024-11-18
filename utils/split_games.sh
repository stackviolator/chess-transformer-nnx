#!/bin/bash

# Check if the input file is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

# Input file
INPUT_FILE="$1"

# Output directory
OUTPUT_DIR="./data/games"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Output file prefix
OUTPUT_PREFIX="$OUTPUT_DIR/lichess_chunk"

# Chunk size in MB
CHUNK_SIZE_MB=50

# Calculate chunk size in bytes
CHUNK_SIZE=$((CHUNK_SIZE_MB * 1024 * 1024))

# Initialize variables
current_size=0
chunk_index=1
output_file="${OUTPUT_PREFIX}_${chunk_index}.pgn"

# Create the first output file
> "$output_file"

# Read the input file line by line
while IFS= read -r line; do
    # Calculate the size of the line in bytes
    line_size=${#line}

    # Check if adding this line exceeds the chunk size
    if (( current_size + line_size > CHUNK_SIZE )); then
        # If we encounter a blank line (game separator), start a new chunk
        if [[ -z "$line" ]]; then
            # Finalize the current chunk
            echo "" >> "$output_file"

            # Move to the next chunk
            chunk_index=$((chunk_index + 1))
            output_file="${OUTPUT_PREFIX}_${chunk_index}.pgn"
            > "$output_file"
            echo "Created chunk file: $output_file"
            current_size=0
            continue
        fi
    fi

    # Write the line to the current output file
    echo "$line" >> "$output_file"

    # Update the current size
    current_size=$((current_size + line_size + 1))

done < "$INPUT_FILE"

