import chess
import flax.nnx as nnx
import jax.numpy as jnp
from src.model.Transformer import Transformer, TransformerConfig
from src.tokenizer.tokenizer import ChessTokenizer
from src.sampler.Sampler import ChessSampler
import warnings

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
    moves = ['<|startofgame|>', 'e4']
    board.push(board.parse_san(moves[-1]))
    illegal_moves = []

    sampler = ChessSampler()

    frequency_penalty = 1.0
    illegal_moves = []

    tokens = tokenizer.encode(moves)

    while True:
        # Break if token length exceeds context length
        if len(tokens) > cfg.ctx_len:
            print("Context length exceeded, exiting.")
            break
        
        # Check combined length of moves and illegal moves
        if len(illegal_moves) + len(moves) > cfg.ctx_len:
            print("Can't find a good move, exiting.")
            break

        # Add empty batch dim for the model
        if len(tokens.shape) == 1:
            tokens = jnp.expand_dims(tokens, 0)

        # Generate logits and sample next move
        logits = model(tokens)
        # pred, pred_tokens = sampler.sample(tokens, logits, tokenizer.encode(illegal_moves), temperature=0.0, frequency_penalty=frequency_penalty)
        # pred, pred_tokens = sampler.sample(tokens, logits, tokenizer.encode(illegal_moves), top_k=5, frequency_penalty=frequency_penalty)
        pred, pred_tokens = sampler.sample(tokens, logits, tokenizer.encode(illegal_moves), top_p=.75, frequency_penalty=frequency_penalty)
        print(f"Sampled tokens: {tokenizer.decode(pred_tokens)}")
        print(f"Predicted move: {tokenizer.decode(pred)}")
        print('-'*10)
        pred = jnp.array(pred)
        move_san = tokenizer.decode([pred.item()])[0]

        # End game condition
        if move_san == "<|endofgame|>":
            print("End of game detected.")
            moves.append(move_san)
            break

        try:
            # Try to parse and apply the move
            move = board.parse_san(move_san)
            board.push(move)
            moves.append(move_san)
            tokens = jnp.append(tokens, pred.item())
            print(f"Moves: {moves}")

            # Reset penalties and illegal moves
            frequency_penalty = 1.0
            illegal_moves = []
        except chess.IllegalMoveError:
            print(f"Illegal move: {move_san}. Retrying...")
            illegal_moves.append(move_san)
            frequency_penalty += len(illegal_moves)
        except chess.AmbiguousMoveError:
            print(f"Ambiguous move: {move_san}. Retrying...")
            illegal_moves.append(move_san)
            frequency_penalty += len(illegal_moves)

    # Output final moves
    print("Final moves:", moves)
    print("Move sequence:", " ".join(m for m in moves))