import csv
import jax.numpy as jnp
import mmap
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import Subset
from tqdm import tqdm


# Lots of inspo taken from https://github.com/codyjk/ChessGPT/blob/main/src/chess_model/data/dataset.py -- thank you :^)
class GamesDataset(Dataset):
    def __init__(self, filename: str, tokenizer, context_length=256):
        self.filename = filename
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.line_offsets = []
        self.file = open(self.filename, "r")

        with open(self.filename, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            total_size = mm.size()
            self.line_offsets.append(0)

            with tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Indexing CSV file"
            ) as pbar:
                while mm.readline():
                    current_pos = mm.tell()
                    self.line_offsets.append(current_pos)
                    pbar.update(current_pos - pbar.n)

            mm.close()

            self.line_offsets.pop()

    def __len__(self):
        return len(self.line_offsets) - 1

    def __getitem__(self, idx) -> dict:
        # Add 1 to idx to skip the header
        items = {
            "input_ids": jnp.empty((1, self.context_length),dtype=jnp.int32), # [batch x context_length]
            "labels": jnp.empty((1, self.context_length), dtype=jnp.int32), # [batch x context_length]
            "is_checkmate": jnp.empty((1, 1), dtype=jnp.int32),
            "outcome": jnp.empty((1,3), dtype=jnp.int32),
            "move_mask": jnp.empty((1, self.context_length), dtype=jnp.int32),
        }
        if isinstance(idx, int):
            idx = [idx]

        for i in idx:
            """
            this is a super hack bc im trying to get a line that doesnt exist
            on second thought, dont think this is super hacky since we need to + 1 to skip the label row
            but this means that if i == len(self.line_offsets) when its + 1'd it will be oob, so decrement just that 
            """
            ## TODO fix me plsss
            if i == len(self.line_offsets) - 1:
                i -= 1
            self.file.seek(self.line_offsets[i + 1])
            line = self.file.readline().strip()

            # Parse the CSV line
            row = next(csv.reader([line]))
            context, is_checkmate, outcome = row

            context = context.split() if context else []
            context, last_move = context[:-1], context[-1]
            is_checkmate = jnp.array(jnp.expand_dims(float(is_checkmate == "True"), axis=0), dtype=jnp.int32)

            input_ids = self.tokenizer.encode_and_pad(context, self.context_length)

            # Shift context to the left to create labels
            # The next move prediction for input_ids[n] is labels[n]
            labels = context[1:] + [last_move]
            labels = self.tokenizer.encode_and_pad(labels, self.context_length)

            # If white won, we want the model to learn from white's moves, not black's.
            # Conversely, if black won, we want the model to learn from black's moves.
            # For draws, we want the model to learn from both moves.
            # We will produce a mask that masks out the moves for the losing player,
            # and the model will learn from the remaining moves.
            move_mask = jnp.ones(self.context_length, dtype=jnp.int32)

            if outcome == "1-0":  # White won
                # Mask out odd-indexed moves (Black's moves)
                move_mask = move_mask.at[1::2].set(0.0)
            elif outcome == "0-1":  # Black won
                # Mask out even-indexed moves (White's moves)
                move_mask = move_mask.at[::2].set(0.0)
            # For draws (1/2-1/2), keep all moves (mask stays 1)

            # If the context is shorter than max_context_length, zero-out that part of the mask
            if len(context) < self.context_length:
                move_mask = move_mask.at[len(context) :].set(0.0)

            # Convert outcome to one-hot encoding (as float)
            outcome_label = jnp.zeros(3, dtype=jnp.int32)
            if outcome == "1-0":
                outcome_label = outcome_label.at[0].set(1.0)
            elif outcome == "0-1":
                outcome_label = outcome_label.at[1].set(1.0)
            elif outcome == "1/2-1/2":
                outcome_label = outcome_label.at[2].set(1.0)

            items["input_ids"] = jnp.concat((items["input_ids"], jnp.array(jnp.expand_dims(input_ids, axis=0), dtype=jnp.int32))) # expand dims to add empty batch dim
            items["labels"] = jnp.concat((items["labels"], jnp.array(jnp.expand_dims(input_ids, axis=0), dtype=jnp.int32)))
            items["is_checkmate"] = jnp.concat((items["is_checkmate"], jnp.array(jnp.expand_dims(is_checkmate, axis=1), dtype=jnp.int32)))
            items["outcome"] = jnp.concat((items["outcome"], jnp.expand_dims(outcome_label, axis=0)))
            items["move_mask"] = jnp.concat((items["move_mask"], jnp.expand_dims(move_mask, axis=0)))

        # kinda (very) hacky, removes the first dim since it was the empty dim when initialized
        return {
        "input_ids": items["input_ids"][1:],
        "labels": items["labels"][1:],
        "is_checkmate": items["is_checkmate"][1:],
        "outcome": items["outcome"][1:],
        "move_mask": items["move_mask"][1:],
    }

    def __del__(self):
        # Close the file when the dataset object is destroyed
        if hasattr(self, "file"):
            self.file.close() 

    def train_test_split(self, test_size: float = 0.2, random_state: int = 1234) -> dict:
        train_indicies, test_indicies = train_test_split(
            range(len(self.line_offsets)),
            test_size=test_size,
            random_state=random_state
        )
        train_dataset = Subset(self, train_indicies)
        test_dataset = Subset(self, test_indicies)
        return {
            "train": train_dataset,
            "test": test_dataset
        }

    @staticmethod
    def collate_fn(batch):
        out = batch[0]
        for d in batch[1:]:
            for k in d.keys():
                out[k] = jnp.concat((out[k], d[k]))
        return out