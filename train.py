from dataclasses import dataclass
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
        print(config_data)
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
    if args.debug == False:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)
    accuracy = jnp.nan
    total_steps = args.epochs * args.max_steps_per_epoch
    step = 0

    with tqdm(total=total_steps, desc="Training Epochs") as progress_bar:
        for epoch in range(args.epochs):
            for i, batch in enumerate(train_loader):
                loss = training_step(model, optimizer, batch)
                if not args.debug:
                    wandb.log({"train_loss":float(loss)}, step=step)
                progress_bar.update()
                progress_bar.set_description(f"Epoch {epoch+1}, loss: {float(loss):.3f}, accuracy: {float(accuracy):.2f}")
                if i >= args.max_steps_per_epoch:
                    break
            correct_sum = 0
            total_count = 0
            for batch in test_loader:
                correct_sum += jnp.sum(validation_step(model, batch))
                total_count += jnp.size(batch["input_ids"]) - 1
            accuracy = correct_sum / total_count
            if not args.debug:
                wandb.log({"accuracy":float(accuracy)}, step=step)

    if args.debug == False:
        wandb.finish()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Tokenizer
    tokenizer = ChessTokenizer()
    tokenizer.load_tokenizer("src/tokenizer/vocab.json")


    pad_token_id = int(tokenizer.encode(["[PAD]"])[0])

    cfg = TransformerConfig.from_yaml('configs/transformer_dev.cfg')
    cfg.d_vocab = len(tokenizer.tokens.values())
    args = TransformerTrainingArgs.from_yaml('configs/training_args.cfg')

    model = Transformer(cfg)

    # Dataset and loaders
    train_file = 'data/clean/games_data.csv'
    dataset = GamesDataset(train_file, tokenizer, context_length=cfg.ctx_len)
    dataset_dict = dataset.train_test_split(test_size=1000)

    train_dataset = dataset_dict["train"]
    test_dataset = dataset_dict["test"]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=GamesDataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=GamesDataset.collate_fn)

    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=args.lr, weight_decay=args.weight_decay))

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
    test_model = model.load(cfg.ckpt_dir)