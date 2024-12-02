import jax.numpy as jnp
import flax.nnx as nnx

class ChessSampler:
    def __init__(self):
        pass

    def greedy(self, logits: jnp.ndarray, illegal_moves: int):
        for idx in illegal_moves:
            logits = logits.at[0, -1, idx].set(float('-inf'))
        probs = nnx.softmax(logits[0,-1,:])
        top_pred = jnp.argmax(probs)
        return [top_pred]
    
    def top_k(self, logits: jnp.ndarray):
        pass

    def top_p(self, logits: jnp.ndarray):
        pass

    def beam_search(self, logits: jnp.ndarray):
        pass