"""Hooks to capture attention weights from DDiTBlock layers during inference.

Temporarily patches the attention function to record Q, K and compute
attention weight matrices, without changing the model's output.
"""

import math
from contextlib import contextmanager

import torch
import einops


class AttentionCapture:
    """Captures attention weights from specified DDiTBlock layers.

    Usage:
        cap = AttentionCapture(model.backbone.blocks, layers=[0, 5, 11])
        with cap.active():
            logp = model.forward(...)
        weights = cap.get_weights()  # dict: layer_idx -> (B, n_heads, L, L)
    """

    def __init__(self, blocks, layers=None):
        """
        blocks: nn.ModuleList of DDiTBlock
        layers: list of int layer indices to capture (None = all)
        """
        self.blocks = blocks
        self.layers = layers if layers is not None else list(range(len(blocks)))
        self._weights = {}
        self._handles = []

    def _make_hook(self, layer_idx):
        capture = self

        def hook_fn(module, inputs, output):
            # Re-derive Q, K from the block's stored intermediate.
            # The block computes: norm -> adaLN -> attn_qkv -> split+RoPE -> attention
            # We intercept by re-running the QKV projection on the same input.
            # This is the input to the block (x before residual).
            x = inputs[0]
            rotary_cos_sin = inputs[1]
            c = inputs[2] if len(inputs) > 2 else None

            x_normed = module.norm1(x)
            if module.adaLN and c is not None:
                chunks = module.adaLN_modulation(c)[:, None].chunk(6, dim=2)
                shift_msa, scale_msa = chunks[0], chunks[1]
                x_normed = x_normed * (1 + scale_msa) + shift_msa

            qkv = einops.rearrange(
                module.attn_qkv(x_normed),
                "b s (three h d) -> b s three h d",
                three=3,
                h=module.n_heads,
            )
            from models.dit_cont import split_and_apply_rotary_pos_emb, _apply_rotary_emb_torch
            q, k, _v = split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin)

            # q, k: (B, S, n_heads, head_dim)
            q_t = q.transpose(1, 2).float()  # (B, n_heads, S, head_dim)
            k_t = k.transpose(1, 2).float()
            head_dim = q_t.shape[-1]
            attn = torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(head_dim)
            attn = attn.softmax(dim=-1)  # (B, n_heads, S, S)
            capture._weights[layer_idx] = attn.detach().cpu()

        return hook_fn

    @contextmanager
    def active(self):
        """Context manager that registers hooks, yields, then removes them."""
        self._weights.clear()
        self._handles.clear()
        for idx in self.layers:
            h = self.blocks[idx].register_forward_hook(self._make_hook(idx))
            self._handles.append(h)
        try:
            yield self
        finally:
            for h in self._handles:
                h.remove()
            self._handles.clear()

    def get_weights(self):
        """Returns dict: layer_idx -> (B, n_heads, L, L) attention weights on CPU."""
        return self._weights

    def clear(self):
        self._weights.clear()
