import unittest
import jax.numpy as jnp
from jax import random
import jax.nn as nnx
from src.model.Trainer import TransformerTrainingArgs, TransformerTrainer
from src.model.Transformer import Transformer, TransformerConfig
from src.tokenizer.tokenizer import ChessTokenizer
from src.dataset.GamesDataset import GamesDataset
from torch.utils.data import DataLoader
import torch as t
from transformer_lens import HookedTransformer

device = t.device('mps' if t.backends.mps.is_available() else 'cuda' if t.cuda.is_available() else 'cpu')

cfg = TransformerConfig(
    d_model=768,
    n_layers=12,
    n_heads=12,
    ln_eps=1e-5,
    d_vocab=50257,
    ctx_len=1024,
    stddev=0.02,
    d_head=64,
    d_mlp=3072,
)
reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,
    device=device
)

# Sample GPT, text, and logits
reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text).to(device)
logits, cache = reference_gpt2.run_with_cache(tokens, device=device)

def get_log_probs(logits, tokens):
    log_probs = logits.log_softmax(dim=-1)
    # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

    return log_probs_for_tokens

class TestGetLogProbs(unittest.TestCase):
    def setUp(self):
        # Tokenizer
        tokenizer = ChessTokenizer()
        tokenizer.load_tokenizer("src/tokenizer/vocab.json")

        # The model config and model itself
        cfg = TransformerConfig(
            d_model=768,
            d_vocab=1974,
            d_head=64,
            d_mlp=3072,
            n_heads=12,
            n_layers=12
        )

        # Traning args
        args = TransformerTrainingArgs(
            epochs=15,
            max_steps_per_epoch=500,
            debug=False,
            )

        transformer = Transformer(cfg)

        # Dataset and loaders
        dataset = GamesDataset("chess_games.csv", tokenizer, context_length=128)
        dataset_dict = dataset.train_test_split(test_size=1000)

        train_dataset = dataset_dict["train"]
        test_dataset = dataset_dict["test"]

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=GamesDataset.collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=GamesDataset.collate_fn)


        # Train the model
        self.trainer = TransformerTrainer(args, transformer, train_loader=train_loader, test_loader=test_loader)

    def test_log_probs(self):
        jax_log_probs = self.trainer.get_log_probs(logits=jnp.array(logits.detach().cpu()), tokens=jnp.array(tokens.detach().cpu()))
        torch_log_probs = jnp.array(get_log_probs(logits=logits, tokens=tokens).detach().cpu())

        self.assertTrue(True and jnp.isclose(jax_log_probs, torch_log_probs).all())