# Hi3D v1.0 (SPARC3D) sparse flow transformer, adapted from Microsoft TRELLIS (MIT) and SPARC3D.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_prefetch
import comfy.ops
import comfy.quant_ops
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.genmo.joint_model.layers import TimestepEmbedder
from .vae import SparseConv3d
from comfy.ldm.modules.attention import ComfyAttention
from comfy.ldm.trellis2.model import MultiHeadRMSNorm, dense_attention
from comfy.ldm.trellis2.vae import SparseTensor


def span_attention(q, k, v, layout, **kwargs):
    """q is [N, H, D] split into batch spans by layout; k/v are [N, H, D] (self) or [B, T, H, D] (cross)."""
    lengths = {s.stop - s.start for s in layout}
    sparse_kv = k.ndim == 3
    if len(lengths) == 1:
        shape = (len(layout), lengths.pop()) + q.shape[1:]
        if sparse_kv:
            k, v = k.view(shape), v.view(shape)
        return dense_attention(q.view(shape), k, v, **kwargs).flatten(0, 1)
    out = []
    for i, s in enumerate(layout):
        ki, vi = (k[s][None], v[s][None]) if sparse_kv else (k[i:i + 1], v[i:i + 1])
        out.append(dense_attention(q[s][None], ki, vi, **kwargs)[0])
    return torch.cat(out)


def modulated_norm(x, layout, scale, shift):
    """layer_norm(x) * (1 + scale) + shift for [N, C] tokens, with per-span [B, C] modulation."""
    lengths = {s.stop - s.start for s in layout}
    if len(lengths) == 1:
        return comfy.quant_ops.ck.adaln(x.view(len(layout), -1, x.shape[1]), scale[:, None], shift[:, None], 1e-6).view(x.shape)
    return torch.cat([comfy.quant_ops.ck.adaln(x[s], scale[i], shift[i], 1e-6) for i, s in enumerate(layout)])


def gated_add_(x, h, gate, layout):
    """x += h * gate in place, with per-span [B, C] gates."""
    lengths = {s.stop - s.start for s in layout}
    if len(lengths) == 1:
        x.view(len(layout), -1, x.shape[1]).addcmul_(h.view(len(layout), -1, x.shape[1]), gate[:, None])
        return
    for i, s in enumerate(layout):
        x[s].addcmul_(h[s], gate[i])


def rope_freqs(coords, max_coord, head_dim):
    """flux-layout rotary table [N, head_dim / 2, 2, 2] for [N, 3] coordinates, stretched past the 64-voxel training grid."""
    freq_dim = head_dim // 6
    freqs = 1.0 / 10000 ** (torch.arange(freq_dim, dtype=torch.float32, device=coords.device) / freq_dim)
    if max_coord >= 64:
        train, infer = 64, (64 + max_coord) // 2
        wavelengths = freqs * train / (2 * math.pi)
        ratio = (wavelengths - 1) / 23
        ratio = ratio + (1 - ratio) * train / infer
        ratio = torch.where(wavelengths > 24, 1.0, ratio)
        ratio = torch.where(wavelengths < 1, train / infer, ratio)
        freqs = freqs * ratio
    phases = (coords.float()[:, :, None] * freqs).flatten(1)
    cos, sin = torch.cos(phases), torch.sin(phases)
    return torch.stack((torch.stack((cos, sin), dim=-1), torch.stack((-sin, cos), dim=-1)), dim=-1)


def sparse_downsample(x):
    """Average onto the half-resolution grid. The zero init stays in the mean (scatter_reduce include_self), as in the release."""
    batch, fine = x.coords[:, :1].long(), x.coords[:, 1:].long() // 2
    extent = int(x.spatial_shape[0] + 1) // 2
    code = ((batch * extent + fine[:, :1]) * extent + fine[:, 1:2]) * extent + fine[:, 2:]
    unique, inverse = torch.unique(code.flatten(), return_inverse=True)
    feats = torch.zeros((unique.shape[0], x.feats.shape[1]), device=x.device, dtype=x.dtype)
    feats.index_add_(0, inverse, x.feats)
    counts = torch.bincount(inverse, minlength=unique.shape[0]).add_(1)
    feats /= counts.unsqueeze(1).to(feats.dtype)
    coords = torch.stack((unique // extent ** 3, unique // extent ** 2 % extent, unique // extent % extent, unique % extent), dim=1).int()
    out = SparseTensor(feats, coords, shape=x.shape, scale=(2, 2, 2), spatial_cache=x._spatial_cache)
    out.register_spatial_cache("shape", torch.Size((extent,) * 3))
    out.register_spatial_cache("upsample", inverse)
    return out


def sparse_upsample(x, fine):
    return fine.replace(x.feats[x.get_spatial_cache("upsample")])


class SparseResBlock3d(nn.Module):
    def __init__(self, channels, emb_channels, out_channels, dtype=None, device=None, operations=None):
        super().__init__()
        hidden = out_channels // 2
        self.norm1 = operations.LayerNorm(channels, eps=1e-6, dtype=dtype, device=device)
        self.conv1 = SparseConv3d(channels, hidden, dtype=dtype, device=device)
        self.conv2 = SparseConv3d(hidden, hidden, dtype=dtype, device=device)
        self.linear = operations.Linear(hidden, out_channels, dtype=dtype, device=device)
        self.emb_layers = nn.Sequential(nn.SiLU(), operations.Linear(emb_channels, 2 * hidden, dtype=dtype, device=device))
        self.skip_connection = operations.Linear(channels, out_channels, dtype=dtype, device=device) if channels != out_channels else nn.Identity()

    def forward(self, x, emb):
        scale, shift = self.emb_layers(emb).chunk(2, dim=1)
        h = self.conv1(x.replace(F.silu(self.norm1(x.feats))))
        h = self.conv2(h.replace(F.silu(modulated_norm(h.feats, x.layout, scale, shift), inplace=True)))
        return x.replace(self.linear(F.silu(h.feats, inplace=True)).add_(self.skip_connection(x.feats)))


class SparseSelfAttention(nn.Module):
    def __init__(self, channels, num_heads, dtype=None, device=None, operations=None):
        super().__init__()
        self.num_heads = num_heads
        self.to_qkv = operations.Linear(channels, channels * 3, dtype=dtype, device=device)
        self.q_rms_norm = MultiHeadRMSNorm(channels // num_heads, num_heads, dtype=dtype, device=device)
        self.k_rms_norm = MultiHeadRMSNorm(channels // num_heads, num_heads, dtype=dtype, device=device)
        self.to_out = operations.Linear(channels, channels, dtype=dtype, device=device)
        self.comfy_attention = ComfyAttention()

    def forward(self, x, freqs, layout, transformer_options):
        q, k, v = self.to_qkv(x).view(x.shape[0], 3, self.num_heads, -1).unbind(1)
        q, k = apply_rope(q[None], k[None], freqs)
        h = span_attention(self.q_rms_norm(q[0]), self.k_rms_norm(k[0]), v, layout, transformer_options=transformer_options, preferred_attention=self.comfy_attention)
        return self.to_out(h.flatten(1))


class SparseCrossAttention(nn.Module):
    def __init__(self, channels, ctx_channels, num_heads, dtype=None, device=None, operations=None):
        super().__init__()
        self.num_heads = num_heads
        self.to_q = operations.Linear(channels, channels, dtype=dtype, device=device)
        self.to_kv = operations.Linear(ctx_channels, channels * 2, dtype=dtype, device=device)
        self.to_out = operations.Linear(channels, channels, dtype=dtype, device=device)
        self.comfy_attention = ComfyAttention()

    def forward(self, x, context, layout, transformer_options):
        q = self.to_q(x).view(x.shape[0], self.num_heads, -1)
        k, v = self.to_kv(context).view(context.shape[0], context.shape[1], 2, self.num_heads, -1).unbind(2)
        h = span_attention(q, k, v, layout, transformer_options=transformer_options, preferred_attention=self.comfy_attention)
        if torch.isnan(h).any():
            # attention kernel produced NaN in this dtype (seen on RDNA2/fp16): redo it in fp32
            outs = []
            for i, sp in enumerate(layout):
                qi = q[sp].float().permute(1, 0, 2)[None]
                ki = k[i].float().permute(1, 0, 2)[None]
                vi = v[i].float().permute(1, 0, 2)[None]
                outs.append(F.scaled_dot_product_attention(qi, ki, vi)[0].permute(1, 0, 2))
            h = torch.cat(outs).to(q.dtype)
        return self.to_out(h.flatten(1))


class ModulatedSparseTransformerCrossBlock(nn.Module):
    def __init__(self, channels, ctx_channels, num_heads, mlp_ratio, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm2 = operations.LayerNorm(channels, eps=1e-6, dtype=dtype, device=device)
        self.self_attn = SparseSelfAttention(channels, num_heads, dtype=dtype, device=device, operations=operations)
        self.cross_attn = SparseCrossAttention(channels, ctx_channels, num_heads, dtype=dtype, device=device, operations=operations)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(operations.Linear(channels, hidden, dtype=dtype, device=device), nn.GELU(approximate="tanh"),
                                 operations.Linear(hidden, channels, dtype=dtype, device=device))
        self.modulation = nn.Parameter(torch.empty(6 * channels, dtype=dtype, device=device))

    def forward(self, x, mod, context, freqs, layout, transformer_options):
        """Updates the [N, C] token features x in place."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (comfy.ops.cast_to_input(self.modulation, mod) + mod).chunk(6, dim=1)
        h = self.self_attn(modulated_norm(x, layout, scale_msa, shift_msa), freqs, layout, transformer_options)
        gated_add_(x, h, gate_msa, layout)
        x.add_(self.cross_attn(self.norm2(x), context, layout, transformer_options))
        gated_add_(x, self.mlp(modulated_norm(x, layout, scale_mlp, shift_mlp)), gate_mlp, layout)


class SparseFlowTransformer(nn.Module):
    def __init__(self, in_channels=32, out_channels=32, io_channels=256, model_channels=2112, cond_channels=2048,
                 num_blocks=24, num_heads=22, mlp_ratio=4.0, dtype=None, device=None, operations=None):
        super().__init__()
        self.head_dim = model_channels // num_heads
        self.t_embedder = TimestepEmbedder(model_channels, dtype=dtype, device=device, operations=operations)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), operations.Linear(model_channels, 6 * model_channels, dtype=dtype, device=device))
        self.input_layer = operations.Linear(in_channels, io_channels, dtype=dtype, device=device)
        args = {"dtype": dtype, "device": device, "operations": operations}
        self.input_blocks = nn.ModuleList([
            SparseResBlock3d(io_channels, model_channels, io_channels, **args),
            SparseResBlock3d(io_channels, model_channels, model_channels, **args),
        ])
        self.blocks = nn.ModuleList([ModulatedSparseTransformerCrossBlock(model_channels, cond_channels, num_heads, mlp_ratio, **args) for _ in range(num_blocks)])
        self.out_blocks = nn.ModuleList([
            SparseResBlock3d(model_channels * 2, model_channels, io_channels, **args),
            SparseResBlock3d(io_channels * 2, model_channels, io_channels, **args),
        ])
        self.out_layer = operations.Linear(io_channels, out_channels, dtype=dtype, device=device)

    def forward(self, x, timestep, context, max_coord, transformer_options):
        t_emb = self.t_embedder(timestep, out_dtype=x.dtype)
        mod = self.adaLN_modulation(t_emb)
        fine = x
        h = x.replace(self.input_layer(x.feats))
        h = self.input_blocks[0](h, t_emb)
        skips = [h.feats]
        h = self.input_blocks[1](sparse_downsample(h), t_emb)
        skips.append(h.feats)

        freqs = rope_freqs(h.coords[:, 1:], max_coord // 2, self.head_dim)[None, :, None]
        layout = h.layout
        feats = h.feats.clone()
        device = feats.device
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), device, transformer_options)
        comfy.model_prefetch.malloc_graph_begin(device)
        for block in self.blocks:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, feats.dtype, malloc_scope="block")
            block(feats, mod, context, freqs, layout, transformer_options)
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None, malloc_scope="block")
        comfy.model_prefetch.malloc_graph_end()

        h = self.out_blocks[0](sparse_upsample(h.replace(torch.cat((feats, skips[1]), dim=1)), fine), t_emb)
        h = self.out_blocks[1](h.replace(torch.cat((h.feats, skips[0]), dim=1)), t_emb)
        return self.out_layer(F.layer_norm(h.feats, h.feats.shape[-1:]))


class Hi3D(nn.Module):
    """transformer_512 runs the 64^3 structure pass, transformer_1024 the 128^3 and 192^3 passes."""
    def __init__(self, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.transformer_512 = SparseFlowTransformer(dtype=dtype, device=device, operations=operations)
        self.transformer_1024 = SparseFlowTransformer(dtype=dtype, device=device, operations=operations)

    def forward(self, x, timestep, context, hi3d_coords, hi3d_coord_counts, hi3d_resolution, transformer_options={}, **kwargs):
        """x is the padded sparse latent [B, C, N, 1]; hi3d_coords [sum(counts), 4] holds (item, x, y, z) for each latent item."""
        B, C, N, _ = x.shape
        counts = hi3d_coord_counts
        lengths = [counts[b % len(counts)] for b in range(B)]
        max_coord = int(hi3d_coords[:, 1:].max())
        coords = hi3d_coords.to(x.device, torch.int32)
        coords = torch.cat([coords + torch.tensor([r * len(counts), 0, 0, 0], dtype=torch.int32, device=x.device) for r in range(B // len(counts))])
        feats = x[..., 0].transpose(1, 2)
        padded = any(length != N for length in lengths)
        feats = torch.cat([feats[b, :length] for b, length in enumerate(lengths)]) if padded else feats.reshape(B * N, C)

        sparse = SparseTensor(feats, coords, shape=torch.Size([B, C]))
        sparse.register_spatial_cache("shape", torch.Size((hi3d_resolution,) * 3))
        starts = [sum(lengths[:b]) for b in range(B)]
        sparse.register_spatial_cache("layout", [slice(start, start + length) for start, length in zip(starts, lengths)])
        model = self.transformer_512 if hi3d_resolution <= 64 else self.transformer_1024
        out = model(sparse, timestep, context, max_coord, transformer_options)

        if not padded:
            return out.view(B, N, -1).transpose(1, 2).unsqueeze(-1)
        result = out.new_zeros((B, N, out.shape[1]))
        for b, (start, length) in enumerate(zip(starts, lengths)):
            result[b, :length] = out[start:start + length]
        return result.transpose(1, 2).unsqueeze(-1)
