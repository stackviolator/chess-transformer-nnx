import jax
import jax.numpy as jnp
import flax.nnx as nnx
from datetime import datetime

class ChessSampler:
    def __init__(self, seed=None):
        self.key = jax.random.PRNGKey(seed) if seed is not None else jax.random.PRNGKey(int(datetime.now().timestamp()))

    def sample(self, tokens:jnp.ndarray, logits: jnp.ndarray, illegal_moves: list=None, temperature=1.0, top_k=0, top_p=0.0, frequency_penalty=5.0):
        if frequency_penalty != 0.0:
            self.apply_frequency_penalty(tokens, logits, frequency_penalty)

        for idx in illegal_moves:
            logits = logits.at[0, -1, idx].set(float('-inf'))

        if temperature == 0.0:
            return self.sample_greedy(logits)
        elif temperature != 1.0:
            logits = self.apply_temperature(logits, temperature)
        if top_k > 0:
            return self.sample_top_k(logits, top_k)
        if top_p > 0.0:
            return self.sample_top_p(logits, top_p)
        
        return self.sample_basic(logits)

    def apply_temperature(self, logits: jnp.ndarray, temperature: float):
        return logits / temperature
    
    def apply_frequency_penalty(self, tokens: jnp.ndarray, logits: jnp.ndarray, freq_penalty: float):
        (_, _, vocab_size) = logits.shape
        count = jnp.bincount(jnp.squeeze(tokens), minlength=vocab_size)
        return logits - freq_penalty * count

    def sample_greedy(self, logits: jnp.ndarray):
        probs = nnx.softmax(logits[0,-1,:])
        top_pred = jnp.argmax(probs)
        return [top_pred], [top_pred]
    
    def sample_top_k(self, logits: jnp.ndarray, k: int):
        probs = nnx.softmax(logits[0,-1,:])
        values, indicies = jax.lax.top_k(probs, k)
        idx = jax.random.categorical(self.key, logits=values)
        return [indicies[idx]], indicies

    def sample_top_p(self, logits: jnp.ndarray, p: float, min_keep: int = 1):
        indicies = jnp.argsort(logits, descending=True)
        logits = jnp.take_along_axis(logits, indicies, axis=-1)
        cumul_probs = jnp.cumsum(nnx.softmax(logits, axis=-1), axis=-1)
        n_keep = jnp.searchsorted(cumul_probs[-1,-1], p, side="right")
        n_keep = max(n_keep, min_keep)
        keep_indicies = indicies[-1, -1, :n_keep]
        keep_logits = logits[-1, -1, :n_keep]
        idx = jax.random.categorical(self.key, logits=keep_logits)
        return [keep_indicies[idx]], keep_indicies


    def sample_basic(self, logits: jnp.ndarray):
        pass