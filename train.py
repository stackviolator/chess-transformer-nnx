import argparse
from dataclasses import dataclass, asdict
from flax import nnx
import jax.numpy as jnp
import jax
import optax
import orbax.checkpoint as ocp
from src.model.Transformer import Transformer, TransformerConfig
from src.tokenizer.tokenizer import ChessTokenizer
from src.dataset.GamesDataset import GamesDataset
import sys
from torch.utils.data import DataLoader
import traceback
from tqdm import tqdm
import wandb
import warnings
import yaml

@dataclass
class TransformerTrainingArgs():
    batch_size: int = 16
    epochs: int = 10
    max_steps_per_epoch: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-2
    wandb_project: str | None = "ChessTransformer"
    wandb_name: str | None = None
    debug: bool = False

    @staticmethod
    def from_yaml(filepath: str):
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        config_data = data.get('training_args', {})
        return TransformerTrainingArgs(**config_data)

@nnx.jit
def training_step(model, optimizer: nnx.Optimizer, batch: dict) -> jnp.ndarray:
    def loss_fn(model: Transformer, batch: dict):
        y_pred = model(batch["input_ids"])
        # One hot encode the labels
        labels = jax.nn.one_hot(batch["labels"], model.cfg.d_vocab)
        # Create mask -- masks losing player's moves and pad tokens
        pad_mask = batch["move_mask"] != model.cfg.pad_token_id
        mask = batch["move_mask"] & pad_mask

        log_probs = optax.softmax_cross_entropy(logits=y_pred, labels=labels)
        masked_log_probs = jnp.where(mask, log_probs, 0.0)
        return jnp.sum(masked_log_probs) / jnp.sum(mask)
    
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, batch)

    loss = jax.block_until_ready(loss)
    optimizer.update(grads)
    return loss

@nnx.jit
def validation_step(model, batch: dict) -> jnp.ndarray:
    tokens = batch["input_ids"]
    logits = model(tokens)[:,:-1]
    pred_tokens = jnp.argmax(logits, axis=-1)
    correct = (pred_tokens == tokens[:, 1:]).flatten()

    return correct

def train(model, optimizer):
    if training_args_config.debug == False:
        wandb.init(project=training_args_config.wandb_project, name=training_args_config.wandb_name, config=training_args_config)
    accuracy = jnp.nan
    total_steps = training_args_config.epochs * training_args_config.max_steps_per_epoch
    step = 0

    with tqdm(total=total_steps, desc="Training Epochs") as progress_bar:
        for epoch in range(training_args_config.epochs):
            for i, batch in enumerate(train_loader):
                loss = training_step(model, optimizer, batch)
                if not training_args_config.debug:
                    wandb.log({"train_loss":loss}, step=step)
                progress_bar.update()
                progress_bar.set_description(f"Epoch {epoch+1}, loss: {loss:.3f}, accuracy: {accuracy:.2f}")
                step += 1
                if i >= training_args_config.max_steps_per_epoch:
                    break
            correct_sum = 0
            total_count = 0
            for batch in test_loader:
                correct_sum += jnp.sum(validation_step(model, batch))
                total_count += jnp.size(batch["input_ids"]) - 1
            accuracy = correct_sum / total_count
            if not training_args_config.debug:
                wandb.log({"accuracy":accuracy}, step=step)
            checkpointer = model.async_save(epoch, training_args_config.debug)

        print("Waiting for checkpointer to finish saving model...")
        checkpointer.wait_until_finished()

    if training_args_config.debug == False:
        wandb.finish()

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Chess Transformer Trainer")
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='configs/transformer_dev.cfg',
        help="Path to the transformer configuration file"
    )
    parser.add_argument(
        '-a',
        '--training_args',
        type=str,
        default='configs/training_args.cfg',
        help="Path to the training arguments file"
    )
    parser.add_argument(
        '-t',
        '--tokenizer',
        type=str,
        default='src/tokenizer/vocab.json',
        help="Path to the tokenzier vocab file"
    )
    parser.add_argument(
        '-d',
        '--debug',
        action='store_true',
        help="Enable debugging mode"
    )
    return parser.parse_args()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    args = parse_args()

    # Tokenizer
    tokenizer = ChessTokenizer()
    tokenizer.load_tokenizer(args.tokenizer)

    pad_token_id = int(tokenizer.encode(["[PAD]"])[0])

    transformer_config = TransformerConfig.from_yaml(args.config)
    training_args_config = TransformerTrainingArgs.from_yaml(args.training_args)

    if args.debug:
        transformer_config.debug = args.debug
        training_args_config.debug = args.debug

    # Update config values based on tokenizer
    transformer_config.d_vocab = len(tokenizer.tokens.values())
    transformer_config.pad_token_id = pad_token_id

    model = Transformer(transformer_config)

    # Dataset and loaders
    train_file = 'data/clean/games_data.csv'
    dataset = GamesDataset(train_file, tokenizer, context_length=transformer_config.ctx_len)
    dataset_dict = dataset.train_test_split(test_size=1000)

    train_dataset = dataset_dict["train"]
    test_dataset = dataset_dict["test"]

    train_loader = DataLoader(train_dataset, batch_size=training_args_config.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=GamesDataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=training_args_config.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=GamesDataset.collate_fn)

    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=training_args_config.lr, weight_decay=training_args_config.weight_decay))

    try:
        train(model, optimizer)

    except Exception as e:
        print(f"An exception occurred: {e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        formatted_traceback = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        print("\nTraceback (most recent call last):")
        print(formatted_traceback)

    # Save the model
    model.save()

    # Test loading the model
    print("testing load :)...")
    test_model = model.load(transformer_config.ckpt_dir)