import os
import json
import jax.numpy as jnp

spec_tokens = ['[UNK]', '[PAD]', '<|startofgame|>', '<|endofgame|>']
input_file_dir = "./data/clean/"

class ChessTokenizer:
    def __init__(self,
             spec_tokens: list = ['UNK', '<|startofgame|>', '<|endofgame|>'],
             input_file_dir: str = "./data/clean/",
             output_file: str = "./tokenizer/vocab.json"):
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

        for (_, _, files) in os.walk(input_dir):
            for file in files:
                with open(input_dir + file, 'r') as game_file:
                    for line in game_file:
                        new_tok = line.strip()
                        if new_tok not in self.tokens:
                            self.tokens[new_tok] = len(self.tokens)

                game_file.close()

        for tok in spec_tokens:
            self.tokens[tok] = len(self.tokens)

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
            seq = seq.at[i].set(int(self.tokens[move]))

        return seq

    def encode_and_pad(self, moves: list, context_length: int) -> jnp.ndarray:
        # TODO write unittest if the context_length - length is 0 (lenth = ids.shape[0])
        if len(moves) >= context_length - 2: # need to make space for start and end of game tokens
            moves = moves [:context_length-2]
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
    test_game = ["<|startofgame|>", "e2e4", "c7c5", "g1f3", "d7d6", "f1b5", "c8d7", "d1e2", "g8f6", "b2b3", "e7e6", "c1b2", "f8e7", "e4e5", "d6e5", "f3e5", "e8g8", "e1g1", "a7a6", "e5d7", "b8d7", "b5d3", "b7b5", "a2a4", "c5c4", "b3c4", "b5b4", "d3e4", "a8b8", "d2d3", "a6a5", "b1d2", "d7c5", "b2e5", "b8b6", "e4f3", "d8d7", "d2b3", "b6a6", "b3c5", "e7c5", "d3d4", "c5e7", "f1d1", "f8c8", "c4c5", "a6a7", "e2b5", "f6d5", "b5d7", "a7d7", "d1d3", "f7f6", "f3g4", "g8f7", "e5g3", "f6f5", "g4h5", "g7g6", "h5f3", "e7f6", "g3d6", "d7d6", "c5d6", "c8c2", "f3d5", "e6d5", "a1e1", "c2c6", "d6d7", "c6d6", "e1e8", "d6d7", "e8a8", "f6d8", "g2g3", "f7e6", "a8a6", "e6f7", "g1g2", "g6g5", "g2f3", "d8c7", "a6a7", "g5g4", "f3g2", "f7e6", "a7b7", "e6d6", "b7b5", "d7e7", "g2f1", "e7e4", "b5b7", "h7h5", "b7b5", "f5f4", "f2f3", "g4f3", "g3f4", "e4f4", "f1f2", "c7d8", "b5b7", "d8h4", "f2f1", "f3f2", "b7b6", "1-0", "<|endofgame|>"]


    tokenizer = ChessTokenizer()
    # tokenizer.train()
    # tokenizer.save_tokenizer()
    tokenizer.load_tokenizer("./tokenizer/vocab.json")

    ids = tokenizer.encode_and_pad(test_game, 128)
    print(ids)
