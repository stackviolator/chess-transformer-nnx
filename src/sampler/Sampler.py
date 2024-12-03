import jax
import jax.numpy as jnp
import flax.nnx as nnx
from datetime import datetime

class ChessSampler:
    def __init__(self, seed=None):
        self.key = jax.random.PRNGKey(seed) if seed is not None else jax.random.PRNGKey(int(datetime.now().timestamp()))

    def sample(self, tokens:jnp.ndarray, logits: jnp.ndarray, temperature=1.0, top_k=0, top_p=0.0, frequency_penalty=5.0):
        if frequency_penalty != 0.0:
            self.apply_frequency_penalty(tokens, logits, frequency_penalty)
        if temperature == 0.0:
            return self.sample_greedy(logits)
        elif temperature != 1.0:
            logits = self.apply_temperature(logits, temperature)
        if top_k > 0:
            return self.sample_top_k(logits, top_k)
        if top_p > 0.0:
            return self.sample_top_p(logits)
        
        return self.sample_basic(logits)

    def apply_temperature(self, logits: jnp.ndarray, temperature: float):
        return logits / temperature
    
    def apply_frequency_penalty(self, tokens: jnp.ndarray, logits: jnp.ndarray, freq_penalty: float):
        (_, _, vocab_size) = logits.shape
        count = jnp.bincount(jnp.squeeze(tokens), minlength=vocab_size)
        return logits - freq_penalty * count

    def sample_greedy(self, logits: jnp.ndarray):
        probs = nnx.softmax(logits[0,-1,:])
        print
        top_pred = jnp.argmax(probs)
        return [top_pred]
    
    def sample_top_k(self, logits: jnp.ndarray, k: int):
        probs = nnx.softmax(logits[0,-1,:])
        values, indicies = jax.lax.top_k(probs, k)
        print(values)
        print(indicies)
        print("--------")
        idx = jax.random.categorical(self.key, logits=values)
        return [indicies[idx]]

    def sample_top_p(self, logits: jnp.ndarray):
        pass

    def sample_basic(self, logits: jnp.ndarray):
        pass