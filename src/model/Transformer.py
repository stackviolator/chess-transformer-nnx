from dataclasses import dataclass
import einops
from etils import epath
from flax import nnx
import jax.numpy as jnp
import jax
import orbax.checkpoint as ocp
import os
from src.tokenizer.tokenizer import ChessTokenizer
from torch.utils.data import Dataset

@dataclass
class TransformerConfig:
    debug: bool = True
    d_model: int = 768
    d_vocab: int = 1974
    d_head: int = 64
    n_layers: int = 4
    n_heads: int = 4
    ctx_len: int = 256
    stddev: float = 0.02
    ln_eps: float = 1e-5
    d_mlp: int = d_model*4
    pad_token_id: int | None = None
    ckpt_dir: str = "trained_models/checkpoints/"
    
class Transformer(nnx.Module):
    def __init__(self, cfg):
        self.cfg = cfg
        self.embed = Embed(self.cfg)
        self.pos_embed = PosEmbed(self.cfg)
        self.blocks = [TransformerBlock(self.cfg) for _ in range(cfg.n_layers)]
        self.ln_final = LayerNorm(self.cfg)
        self.unembed = Unembed(self.cfg)

    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        resid = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            resid = block(resid)
        logits = self.unembed(self.ln_final(resid))
        return logits
    
    def save(self):
        path = epath.Path(os.getcwd() + '/' + self.cfg.ckpt_dir)
        if path.exists(): path.rmtree()
        print(f"Saving model to {path}/... ", end="")
        _, state = nnx.split(self)

        # NNX docs were not working, using async checkpoint
        checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        checkpointer.save(path, args=ocp.args.StandardSave(state))
        checkpointer.wait_until_finished()
        print("saved!")

    def load(self, filepath):
        path = epath.Path(os.getcwd() + '/' + self.cfg.ckpt_dir)
        checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        print(f"Loading model from {filepath}")
        graphdef, abstract_state = nnx.split(self)

        state_restored = checkpointer.restore(path, abstract_state)
        return nnx.merge(graphdef, state_restored)

class LayerNorm(nnx.Module):
    def __init__(self, cfg: TransformerConfig):
        key = jax.random.PRNGKey(101)
        self.cfg = cfg
        self.d_model = self.cfg.d_model
        self.w = nnx.Param(jax.random.normal(key, (self.d_model))) # [d_model]
        self.b = nnx.Param(jnp.zeros(self.d_model,)) # [d_model]
        self.eps = self.cfg.ln_eps
    
    def __call__(self, residual: jax.Array):
        # resdiual: [batch x len x d_model]
        # Make mean 0 and normalize to have variance 1
        res_mean = jnp.mean(residual, axis=-1, keepdims=True)
        res_std = jnp.sqrt(jnp.var(residual, axis=-1, keepdims=True) + self.eps)
        # Scale with learned weights
        y = (residual - res_mean) / res_std
        # Translate with learned bias
        y = y * self.w + self.b
        
        return y
    
class Embed(nnx.Module):
    def __init__(self, cfg: TransformerConfig):
        key = jax.random.PRNGKey(101)
        self.cfg = cfg
        self.W_E = nnx.Param(jax.random.normal(key, (self.cfg.d_vocab, self.cfg.d_model)) * self.cfg.stddev)

    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        # tokens: [batch length]
        return self.W_E[tokens]
    
class PosEmbed(nnx.Module):
    def __init__(self, cfg: TransformerConfig):
        key = jax.random.PRNGKey(101)
        self.cfg = cfg
        self.W_pos = nnx.Param(jax.random.normal(key, (cfg.ctx_len, cfg.d_model)) * self.cfg.stddev)

    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        # tokens: [batch length]
        batch, length = tokens.shape
        return einops.repeat(self.W_pos[:length], 'length d_model -> batch length d_model', batch=batch)
    
class Attention(nnx.Module):
    def __init__(self, cfg: TransformerConfig, rngs: nnx.Rngs):
        key = rngs.params()
        self.cfg = cfg
        self.W_Q = nnx.Param(jax.random.normal(key, (cfg.n_heads, cfg.d_model, cfg.d_head))) # [num_heads, d_model, d_head]
        self.W_K = nnx.Param(jax.random.normal(key, (cfg.n_heads, cfg.d_model, cfg.d_head))) # [num_heads, d_model, d_head]
        self.W_V = nnx.Param(jax.random.normal(key, (cfg.n_heads, cfg.d_model, cfg.d_head))) # [num_heads, d_model, d_head]
        self.W_O = nnx.Param(jax.random.normal(key, (cfg.n_heads, cfg.d_head, cfg.d_model))) # [num_heads, d_head, d_model]
        self.b_Q = nnx.Param(jnp.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nnx.Param(jnp.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nnx.Param(jnp.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nnx.Param(jnp.zeros((cfg.d_model)))

    def __call__(self, normal_pre_resid: jnp.ndarray) -> jnp.ndarray:
        """
        b = batch
        l = length
        m = d_model
        n = num_heads
        h = d_head
        q = q_pos
        k = k_pos
        """
        # normal_pre_resid: [batch length d_model]
        q = jnp.einsum('blm, nmh -> blnh', normal_pre_resid, self.W_Q.value) + self.b_Q
        k = jnp.einsum('blm, nmh -> blnh', normal_pre_resid, self.W_K.value) + self.b_K
        v = jnp.einsum('blm, nmh -> blnh', normal_pre_resid, self.W_V.value) + self.b_V

        attn_scores = jnp.einsum('bqnh, bknh -> bnqk', q, k)
        attn_scores = self.apply_casual_mask(attn_scores / self.cfg.d_head ** 0.5)
        attn_probs = jax.nn.softmax(attn_scores, axis=-1) # [batch x n_heads x q_pos x k_pos]

        # [batch x q_pos x n_heads x d_head]
        z = jnp.einsum('bknh, bnqk -> bqnh', v, attn_probs)

        out = jnp.einsum('bqnh, nhm -> bqm', z, self.W_O.value) + self.b_O
        return out

    def apply_casual_mask(self, attn_scores: jnp.ndarray) -> jnp.ndarray:
        all_ones = jnp.ones((attn_scores.shape[-2], attn_scores.shape[-1]))
        # attn_scores: [batch n_heads q_pos k_pos]
        mask = jnp.triu(all_ones, k=1).astype(bool)
        masked_attn_scores = jnp.where(mask, jax.lax.broadcast(-jnp.inf, attn_scores.shape), attn_scores)
        
        return masked_attn_scores
    
class MLP(nnx.Module):
    def __init__(self, cfg: TransformerConfig):
        key = jax.random.PRNGKey(101)
        self.cfg = cfg
        self.W_in = nnx.Param(jax.random.normal(key, (cfg.d_model, cfg.d_mlp))) # [d_model, d_mlp]
        self.W_out = nnx.Param(jax.random.normal(key, (cfg.d_mlp, cfg.d_model))) # [d_mlp, d_model]
        self.b_in = nnx.Param(jnp.zeros((cfg.d_mlp)))
        self.b_out = nnx.Param(jnp.zeros((cfg.d_model)))

    def __call__(self, normal_resid_mid: jnp.ndarray) -> jnp.ndarray:
        # normal_resid_mid [batch x length x d_model]
        """
        b = batch
        l = length
        m = d_model
        p = d_mlp
        """
        out = jnp.einsum('blm, mp -> blp', normal_resid_mid, self.W_in.value) + self.b_in
        out = nnx.gelu(out)
        out = jnp.einsum('blp, pm -> blm', out, self.W_out.value) + self.b_out
        return out

class TransformerBlock(nnx.Module):
    def __init__(self, cfg: TransformerConfig):
        self.cfg = cfg
        self.ln1 = LayerNorm(self.cfg)
        self.ln2 = LayerNorm(self.cfg)
        self.attn = Attention(self.cfg, rngs=nnx.Rngs(params=0))
        self.mlp = MLP(self.cfg)

    def __call__(self, resid_pre: jnp.ndarray) -> jnp.ndarray:
        resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre
        resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
        return resid_post
    
class Unembed(nnx.Module):
    def __init__(self, cfg: TransformerConfig):
        key = jax.random.PRNGKey(101)
        self.cfg = cfg
        self.W_U = nnx.Param(jax.random.normal(key, (cfg.d_model, cfg.d_vocab)))
        self.b_U = nnx.Param(jnp.zeros(cfg.d_vocab))

    def __call__(self, normal_resid_post: jnp.ndarray) -> jnp.ndarray:
        # normal_resid_post: [batch x length x d_model]
        """
        b = batch
        l = length
        m = d_model
        b = d_vocab
        """
        return jnp.einsum('blm, mv -> blv', normal_resid_post, self.W_U.value) + self.b_U
