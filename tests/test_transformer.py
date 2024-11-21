import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from transformer_lens import HookedTransformer
import torch as t
import unittest
import src.model.Transformer as Transformer

cfg = Transformer.TransformerConfig(
    d_model=768,
    n_layers=12,
    n_heads=12,
    ln_eps=1e-5,
    d_vocab=50257,
    ctx_len=1024,
    stddev=0.02,
    d_head=64,
    d_mlp=3072,
)

device = t.device('mps' if t.backends.mps.is_available() else 'cuda' if t.cuda.is_available() else 'cpu')
reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,
    device=device
)

# Sample GPT, text, and logits
reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text).to(device)
logits, cache = reference_gpt2.run_with_cache(tokens, device=device)

def load_gpt2_test(cls, gpt2_layer, input, debug=False):
    jax_input = jnp.array(input.cpu())
    # conver the pytorch state_dict to jax
    state_dict = gpt2_layer.state_dict()
    for k, v in state_dict.items():
        # this is a hack -- setattr(cls, "attn.W_V", ...) will make a new attrib called "attn.W_V" and not update the W_V attrib of the attn obj
        # i doubt this is a robust solution, but i only need it for these tests .. so whatever
        if debug == True:
            print(k)
        if '.' in k:
             # Split the key into parts for nested attributes
            attrs = k.split('.')
            # Start with the base object
            obj = cls
            # Traverse through the nested attributes except the last one
            for attr in attrs[:-1]:
                # If it's an integer (e.g., "blocks.0"), cast it to int for list indexing
                if attr.isdigit():
                    obj = obj[int(attr)]
                else:
                    obj = getattr(obj, attr)
            # Set the final attribute
            setattr(obj, attrs[-1], nnx.Param(jnp.array(v.cpu(), dtype=jnp.float32)))
        else:
            setattr(cls, k, nnx.Param(jnp.array(state_dict[k].cpu(), dtype=jnp.float32)))
    output = t.from_numpy(np.asarray(cls(jax_input))).to(device)   # convert the jax array to a tensor used later in isclose()
    if isinstance(output, tuple): output = output[0]
    try: reference_output = gpt2_layer(input)
    except: reference_output = gpt2_layer(input, input, input)
    comparison = t.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
    print(f"{comparison.sum()/comparison.numel():.2%} of the values are correct\n")

    if debug == True:
        print(f"jax out: {output[:10]}")
        print(f"torch out: {reference_output[:10]}")

    return comparison

class TestTransformer(unittest.TestCase):
    def test_layernorm(self):
        print(f"Testing layernorm")
        comp = load_gpt2_test(Transformer.LayerNorm(cfg), reference_gpt2.ln_final, cache['resid_post', 11])
        # Ensure all vals in comp are "True"
        self.assertTrue(True and t.all(comp))

    def test_embed(self):
        print(f"Testing embed")
        comp = load_gpt2_test(Transformer.Embed(cfg), reference_gpt2.embed, tokens)
        # Ensure all vals in comp are "True"
        self.assertTrue(True and t.all(comp))

    def test_pos_embed(self):
        print(f"Testing positional embed")
        comp = load_gpt2_test(Transformer.PosEmbed(cfg), reference_gpt2.pos_embed, tokens)
        # Ensure all vals in comp are "True"
        self.assertTrue(True and t.all(comp))

    def test_attention(self):
        print(f"Testing attention")
        comp = load_gpt2_test(Transformer.Attention(cfg, rngs=nnx.Rngs(params=0)), reference_gpt2.blocks[0].attn, cache["normalized", 0, "ln1"])
        # Ensure all vals in comp are "True"
        self.assertTrue(True and t.all(comp))

    def test_mlp(self):
        print(f"Testing MLP")
        comp = load_gpt2_test(Transformer.MLP(cfg), reference_gpt2.blocks[0].mlp, cache["normalized", 0, "ln2"])
        # Ensure all vals in comp are "True"
        self.assertTrue(True and t.all(comp))

    def test_transformer_block(self):
        print(f"Testing transformer block")
        comp = load_gpt2_test(Transformer.TransformerBlock(cfg), reference_gpt2.blocks[0], cache["resid_pre", 0])
        # Ensure all vals in comp are "True"
        self.assertTrue(True and t.all(comp))

    def test_unembed(self):
        print(f"Testing unembed")
        comp = load_gpt2_test(Transformer.Unembed(cfg), reference_gpt2.unembed, cache["ln_final.hook_normalized"])
        # Ensure all vals in comp are "True"
        self.assertTrue(True and t.all(comp))

    def test_transformer(self):
        print(f"Testing transformer")
        comp = load_gpt2_test(Transformer.Transformer(cfg), reference_gpt2, tokens)
        # Ensure all vals in comp are "True"
        self.assertTrue(True and t.all(comp))