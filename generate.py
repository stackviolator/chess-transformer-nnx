import argparse
import chess
import flax.nnx as nnx
import jax.numpy as jnp
from src.model.Transformer import Transformer, TransformerConfig
from src.tokenizer.tokenizer import ChessTokenizer
from src.sampler.Sampler import ChessSampler
import warnings


def parse_args():
    parser = argparse.ArgumentParser(description="Generate predictions using a trained model")
    
    parser.add_argument("-m", "--model-path", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Path to save the generated predictions.")
    parser.add_argument("-t", "--tokenizer", type=str, default="src/tokenizer/vocab.json", help="Path to the tokenizer vocab file.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling.")
    parser.add_argument("-k", "--top-k", type=int, default=0, help="Top K for sampling.")
    parser.add_argument("-p", "--top-p", type=float, default=0.0, help="Top P for sampling.")
    parser.add_argument("-c", "--config", type=str, default="configs/transformer_dev.cfg", help="Path to the transformer configuration file.")
    parser.add_argument('-d','--debug',action='store_true',help="Enable debugging mode")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device to run generation on.")

    return parser.parse_args()

def save_samples(samples, output_path):
    print(f"Saving samples to {output_path}...")
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(sample + "\n")

def generate_moves(model, tokenizer, sampler, transformer_config):
    board = chess.Board()
    moves = ['<|startofgame|>', 'e4']
    board.push(board.parse_san(moves[-1]))
    illegal_moves = []

    frequency_penalty = 1.0
    tokens = tokenizer.encode(moves)

    while True:
        if len(tokens) > transformer_config.ctx_len:
            print("Context length exceeded, exiting.")
            break

        if len(illegal_moves) + len(moves) > transformer_config.ctx_len:
            print("Can't find a good move, exiting.")
            break

        if len(tokens.shape) == 1:
            tokens = jnp.expand_dims(tokens, 0)


        logits = model(tokens)

        pred, pred_tokens = sampler.sample(
            tokens, logits, tokenizer.encode(illegal_moves), temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, frequency_penalty=frequency_penalty
        )
        print(f"Sampled tokens: {tokenizer.decode(pred_tokens)}")
        print(f"Predicted move: {tokenizer.decode(pred)}")
        print('-'*10)
        pred = jnp.array(pred)

        move_san = tokenizer.decode([pred.item()])[0]

        if move_san == "<|endofgame|>":
            print("End of game detected.")
            moves.append(move_san)
            break

        try:
            move = board.parse_san(move_san)
            board.push(move)
            moves.append(move_san)
            tokens = jnp.append(tokens, pred.item())

            frequency_penalty = 1.0
            illegal_moves = []
        except (chess.IllegalMoveError, chess.AmbiguousMoveError):
            print(f"Invalid move: {move_san}. Retrying...")
            illegal_moves.append(move_san)
            frequency_penalty += len(illegal_moves)

    print("Final moves:", moves)
    return moves

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    args = parse_args()

    # Tokenizer
    tokenizer = ChessTokenizer()
    tokenizer.load_tokenizer(args.tokenizer)
    pad_token_id = int(tokenizer.encode(["[PAD]"])[0])

    transformer_config = TransformerConfig.from_yaml(args.config)

    if args.debug:
        transformer_config.debug = args.debug

    transformer_config.d_vocab = len(tokenizer.tokens.values())
    transformer = Transformer(transformer_config)
    model = transformer.load(transformer_config.ckpt_dir)

    sampler = ChessSampler()

    moves = generate_moves(model, tokenizer, sampler, transformer_config)

    # Output final moves
    print("Final moves:", moves)
    print("Move sequence:", " ".join(m for m in moves))
    save_samples(moves, args.output_path)