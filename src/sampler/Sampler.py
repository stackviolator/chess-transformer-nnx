import jax.numpy as jnp
import flax.nnx as nnx

class ChessSampler:
    def __init__(self):
        pass

    def greedy(self, logits: jnp.ndarray):
        probs = nnx.softmax(logits[0,-1,:])
        top_pred = jnp.argmax(probs)
        return [top_pred]