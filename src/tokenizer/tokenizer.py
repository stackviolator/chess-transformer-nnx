import csv
import os
import json
import jax.numpy as jnp

spec_tokens = ['[UNK]', '[PAD]', '<|startofgame|>', '<|endofgame|>']
input_file_dir = "./data/clean/"

class ChessTokenizer:
    def __init__(self,
             spec_tokens: list = ['UNK', '<|startofgame|>', '<|endofgame|>'],
             input_file_dir: str = "data/clean/",
             output_file: str = "src/tokenizer/vocab.json"):
        self.spec_tokens = spec_tokens
        self.input_file_dir = input_file_dir
        self.output_file = output_file
        self.tokens = dict()

    def train(self, input_dir=None):
        """
        Creates the vocab for the tokenizer
        Vocab will be in format {str:int}
        """
        input_dir = self.input_file_dir if input_dir is None else input_dir
        for _, _, files in os.walk(input_dir):
            for file in files:
                file_path = os.path.join(input_dir, file)
                with open(file_path, 'r') as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    for row in csv_reader:
                        moves = row['Moves']
                        for move in moves.split():
                            stripped_move = move.strip()
                            self.tokens.setdefault(stripped_move, len(self.tokens))

        for tok in spec_tokens:
            self.tokens.setdefault(tok, len(self.tokens))

    def save_tokenizer(self, outfile=None):
        """
        Saves vocab to outfile
        """
        outfile = self.output_file if outfile is None else outfile

        with open(outfile, "w") as json_file:
            print(f"Writing tokens to {outfile}")
            json.dump(self.tokens, json_file, indent=4)


    def load_tokenizer(self, filepath=None):
        """
        Loads a tokenizer into self.tokens
        """
        try:
            with open(filepath, "r") as f:
                self.tokens = json.load(f)
        except:
            print(f"[-] Could not open vocab file \"{filepath}\"")

    def invert_tokens(self):
        if not isinstance(self.tokens, dict):
            raise ValueError("self.tokens must be a dictionary")

        # Invert the dictionary
        inverted_tokens = {value: key for key, value in self.tokens.items()}

        return inverted_tokens

    def encode(self, moves: list) -> jnp.ndarray:
        """
        Encode sequence of moves to tokens
        """
        seq = jnp.empty([len(moves)], dtype=int)

        for i, move in enumerate(moves):
            # Check if token exists
            if move in self.tokens.keys():
                seq = seq.at[i].set(int(self.tokens[move]))
            else:
                seq = seq.at[i].set(int(self.tokens['[UNK]']))

        return seq

    def encode_and_pad(self, moves: list, context_length: int) -> jnp.ndarray:
        if len(moves) > context_length:
            moves = moves[:context_length]
        ids = self.encode(moves)

        pad_arr = jnp.repeat(jnp.array([self.tokens["[PAD]"]]), context_length - ids.shape[0]) # TODO: error here when we exceed context len

        ids = jnp.concat([ids, pad_arr])
        return ids


    def decode(self, tokens: jnp.ndarray) -> list:
        """
        Decode tokens to array of moves
        """
        seq = list()
        itokens = self.invert_tokens()

        for tok in tokens:
            seq.append(itokens[int(tok)])

        return seq

if __name__ == "__main__":
    tokenizer = ChessTokenizer()
    tokenizer.train()
    tokenizer.save_tokenizer()