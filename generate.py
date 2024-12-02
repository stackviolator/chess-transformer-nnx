import flax.nnx as nnx
import jax.numpy as jnp
from src.model.Transformer import Transformer, TransformerConfig
from src.tokenizer.tokenizer import ChessTokenizer
from src.sampler.Sampler import ChessSampler
import warnings

test_game = """
<|startofgame|> e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 e6 Nb3 d6 Nc3 Nf6 Be3 Be7 f3 O-O Qd2 a6 O-O-O b5 g4 Bb7 h4 Na5 h5 Nxb3+ axb3 d5 exd5 exd5 Bd4 b4 Ne2 Nd7 Ng3 Bf6 Nf5 Bxd4 Qxd4 Nb6 Qxg7# <|endofgame|>
"""

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Tokenizer
    tokenizer = ChessTokenizer()
    tokenizer.load_tokenizer("src/tokenizer/vocab.json")

    sampler = ChessSampler()

    pad_token_id = int(tokenizer.encode(["[PAD]"])[0])

    # The model config and model itself
    cfg = TransformerConfig(
        d_model=768,
        d_vocab=len(tokenizer.tokens.values()),
        d_head=64,
        d_mlp=3072,
        n_heads=12,
        n_layers=12,
        ctx_len=128,
        pad_token_id=pad_token_id,
        ckpt_dir="trained_models/dev"
    )

    transformer = Transformer(cfg)

    # Train the model
    # Test loading the model
    print(f"Loading model at {cfg.ckpt_dir} ...")
    model = transformer.load(cfg.ckpt_dir)

    moves = ['<|startofgame|>', 'e4']

    for i in range(20):
        tokens = tokenizer.encode(moves)
        tokens = jnp.expand_dims(tokens, 0)
        logits = model(tokens)
        top_pred = sampler.greedy(logits)
        moves.append(tokenizer.decode(top_pred)[0])
        print(moves[-1])