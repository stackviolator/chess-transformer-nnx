import chess
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

    pad_token_id = int(tokenizer.encode(["[PAD]"])[0])

    cfg = TransformerConfig.from_yaml('configs/transformer_dev.cfg')
    cfg.d_vocab = len(tokenizer.tokens.values())

    transformer = Transformer(cfg)

    # Train the model
    # Test loading the model
    print(f"Loading model at {cfg.ckpt_dir} ...")
    model = transformer.load(cfg.ckpt_dir)

    board = chess.Board()
    moves = ['<|startofgame|>', 'd4']
    board.push(board.parse_san(moves[-1]))
    illegal_moves = []

    sampler = ChessSampler()

    frequency_penalty = 1.0
    illegal_moves = []

    while True:
        if len(moves) > cfg.ctx_len:
            break
        if len(illegal_moves) > cfg.ctx_len:
            print("Can't find good move")
            break
        tokens = tokenizer.encode(moves)
        if len(illegal_moves) > 0:
            illegal_tokens = tokenizer.encode(illegal_moves)
            tokens = jnp.concat([tokens, illegal_tokens])
        tokens = jnp.expand_dims(tokens, 0)
        logits = model(tokens)
        pred = sampler.sample(tokens, logits, top_k=5, frequency_penalty=frequency_penalty)
        move_san = tokenizer.decode(pred)[0]
        if move_san == "<|endofgame|>":
            break
        try:
            move = board.parse_san(move_san)
            moves.append(move_san)
            board.push(move)
            frequency_penalty = 1.0
            illegal_moves = []
            print(f"moves: {moves}")
        except chess.IllegalMoveError:
            print(f"Illegal move {move_san} retrying")
            illegal_moves.append(move_san)
            frequency_penalty += len(illegal_moves)
        except chess.AmbiguousMoveError:
            print(f"Ambiguous move {move_san} retrying")
            illegal_moves.append(move_san)
            frequency_penalty += len(illegal_moves)

    print(moves)

print(" ".join(m for m in moves[1:]))