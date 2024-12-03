from dataclasses import dataclass
from flax import nnx
import jax.numpy as jnp
import jax
import optax
import orbax.checkpoint as ocp
from src.model.Transformer import Transformer
from tqdm import tqdm
import wandb

@dataclass
class TransformerTrainingArgs():
    batch_size = 16
    epochs: int = 10
    max_steps_per_epoch: int = 200
    lr = 1e-3
    weight_decay = 1e-2
    wandb_project: str | None = "ChessTransformer"
    wandb_name: str | None = None
    debug: bool = False

class TransformerTrainer:
    def __init__(self, args: TransformerTrainingArgs, model: Transformer, train_loader, test_loader):
        super().__init__()
        self.args = args
        self.optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=args.lr, weight_decay=args.weight_decay))
        self.step = 0
        self.train_loader = train_loader
        self.test_loader = test_loader

    def training_step(self, model, optimizer: nnx.Optimizer, batch: dict) -> jnp.ndarray:
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

        self.step += 1
        if self.args.debug == False:
            wandb.log({"train_loss":float(loss)}, step=self.step)
        if self.args.debug:
            print(f"Loss: {loss}")
        return loss

    def validation_step(self, model, batch: dict) -> jnp.ndarray:
        tokens = batch["input_ids"]
        logits = model(tokens)[:,:-1]
        pred_tokens = jnp.argmax(logits, axis=-1)
        correct = (pred_tokens == tokens[:, 1:]).flatten()

        return correct

    def train(self, model):
        if self.args.debug == False:
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args)
        accuracy = jnp.nan
        total_steps = self.args.epochs * self.args.max_steps_per_epoch

        with tqdm(total=total_steps, desc="Training Epochs") as progress_bar:
            for epoch in range(self.args.epochs):
                for i, batch in enumerate(self.train_loader):
                    loss = self.training_step(model, self.optimizer, batch)
                    if self.args.debug and self.step % 100 == 0:
                        print(f"{i} steps")
                        jax.profiler.save_device_memory_profile(f"/tmp/memory{i}.prof")
                    progress_bar.update()
                    progress_bar.set_description(f"Epoch {epoch+1}, loss: {loss:.3f}, accuracy: {accuracy:.2f}")
                    if i >= self.args.max_steps_per_epoch:
                        break
                correct_sum = 0
                total_count = 0
                for batch in self.test_loader:
                    correct_sum += jnp.sum(self.validation_step(model, batch))
                    total_count += jnp.size(batch["input_ids"]) - 1
                accuracy = correct_sum / total_count
                if not self.args.debug:
                    wandb.log({"accuracy":float(accuracy)}, step=self.step)

        if self.args.debug == False:
            wandb.finish()

    def get_log_probs(self, logits: jnp.ndarray, tokens: jnp.ndarray) -> jnp.ndarray:
        log_probs = nnx.log_softmax(logits, axis=-1)
        sliced_log_probs = log_probs[:, :-1]
        next_token_indicies = jnp.expand_dims(tokens[:, 1:], axis=-1).astype(jnp.int32)
        log_probs_for_tokens = jnp.take_along_axis(
            sliced_log_probs, next_token_indicies, axis=-1
        )

        return jnp.squeeze(log_probs_for_tokens, axis=-1)