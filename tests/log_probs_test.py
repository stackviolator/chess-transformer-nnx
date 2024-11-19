import unittest
import jax.numpy as jnp
from jax import random
import jax.nn as nnx

# NOTE: this test is not real and sucks bc chatgpt wrote it

class TestGetLogProbs(unittest.TestCase):
    def setUp(self):
        # Example function to test
        class DummyModel:
            def get_log_probs(self, logits: jnp.ndarray, tokens: jnp.ndarray) -> jnp.ndarray:
                log_probs = nnx.log_softmax(logits, axis=-1)
                sliced_log_probs = log_probs[:, :-1]
                next_token_indices = jnp.expand_dims(tokens[:, 1:], axis=-1)
                log_probs_for_tokens = jnp.take_along_axis(
                    sliced_log_probs, next_token_indices, axis=-1
                )
                return log_probs_for_tokens

        self.model = DummyModel()

    def test_get_log_probs(self):
        # Test data
        batch_size = 2
        seq_len = 4
        vocab_size = 5

        # Define logits (random) and tokens (deterministic for simplicity)
        key = random.PRNGKey(0)
        logits = random.normal(key, (batch_size, seq_len, vocab_size))
        tokens = jnp.array([[0, 1, 2, 3], [3, 2, 1, 0]])

        # Expected result (manual computation for verification)
        log_probs = nnx.log_softmax(logits, axis=-1)
        sliced_log_probs = log_probs[:, :-1]
        next_token_indices = jnp.expand_dims(tokens[:, 1:], axis=-1)
        expected_log_probs_for_tokens = jnp.take_along_axis(
            sliced_log_probs, next_token_indices, axis=-1
        )

        # Call the method
        result = self.model.get_log_probs(logits, tokens)

        # Assert results
        self.assertTrue(jnp.allclose(result, expected_log_probs_for_tokens),
                        "Computed log probabilities do not match the expected result")

if __name__ == "__main__":
    unittest.main()

