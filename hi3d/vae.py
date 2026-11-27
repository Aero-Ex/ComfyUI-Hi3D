# Hi3D v1.0 (SPARC3D) sparse mesh decoder, adapted from Pointcept / Point Transformer V3 (MIT)
# and SPARC3D. FlexiCubes extraction from NVIDIA FlexiCubes (Apache-2.0).

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops
from .flexicubes import FlexiCubes
from comfy.ldm.modules.attention import ComfyAttention
from comfy.ldm.trellis2.flexgemm import TorchHashMap, build_submanifold_neighbor_map, get_recommended_chunk_mem
from comfy.ldm.trellis2.model import dense_attention
from comfy.ldm.trellis2.vae import SparseTensor

ops = comfy.ops.disable_weight_init

CUBE_CORNERS = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1))


class SparseConv3d(nn.Module):
    """3x3x3 submanifold convolution, weight in (out, kx, ky, kz, in) layout."""
    def __init__(self, in_channels, out_channels, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, 3, 3, 3, in_channels, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype))

    def forward(self, x, norm=None, out=None):
        """conv(norm(x)) with the norm applied to each gathered chunk, so normalized features are never stored for the
        whole shape. Adds into out when given."""
        feats = x.feats
        side = x.spatial_shape[0]
        hashmap = voxel_hashmap(x)
        weight = comfy.ops.cast_to_input(self.weight, feats, copy=False).view(self.weight.shape[0], -1).t()
        if norm is not None:
            scale, shift = norm.affine(feats)
        accumulate = out is not None
        if out is None:
            out = torch.empty(feats.shape[0], weight.shape[1], device=feats.device, dtype=feats.dtype)
        chunk = max(1, int(get_recommended_chunk_mem(feats.device) * 1024 ** 3 / (weight.shape[0] * feats.element_size())))
        for start in range(0, feats.shape[0], chunk):
            index = build_submanifold_neighbor_map(hashmap, x.coords[start:start + chunk], side, side, side, 3, 3, 3, 1, 1, 1)
            gathered = feats[index.clamp_min(0)]
            if norm is not None:
                F.silu(torch.addcmul(shift, gathered, scale, out=gathered), inplace=True)
            gathered.masked_fill_(index[:, :, None] < 0, 0)
            out[start:start + chunk].addmm_(gathered.view(index.shape[0], -1), weight, beta=int(accumulate))
        return x.replace(out.add_(comfy.ops.cast_to_input(self.bias, out, copy=False)))


def voxel_hashmap(x):
    """Grid-key hashmap of the tensor's voxels, cached on it. Convs look neighbors up chunk by chunk because an
    [N, 27] neighbor map does not fit next to the finest stage's features."""
    hashmap = x.get_spatial_cache("hashmap")
    if hashmap is None:
        side = x.spatial_shape[0]
        keys = ((x.coords[:, 0].long() * side + x.coords[:, 1]) * side + x.coords[:, 2]) * side + x.coords[:, 3]
        hashmap = TorchHashMap(keys, torch.arange(keys.shape[0], dtype=torch.int32, device=keys.device))
        x.register_spatial_cache("hashmap", hashmap)
    return hashmap


def sparse_tensor(feats, coords, resolution):
    x = SparseTensor(feats, coords, shape=torch.Size([1, feats.shape[1]]))
    x.register_spatial_cache("shape", torch.Size((resolution,) * 3))
    x.register_spatial_cache("layout", [slice(0, feats.shape[0])])
    return x


class PointGroupNormSiLU(nn.Module):
    """silu(group_norm(x)) over all [N, C] points of one shape. Runs in place, or as an affine inside SparseConv3d's
    gather, because the finest decoder stage has no room for a separate normalized copy."""
    def __init__(self, channels, groups=32, eps=1e-6):
        super().__init__()
        self.groups = groups
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(channels))
        self.bias = nn.Parameter(torch.empty(channels))

    def affine(self, x):
        """Per-channel (scale, shift) of the norm, in x.dtype."""
        grouped = x.view(x.shape[0], self.groups, -1)
        count = grouped.shape[0] * grouped.shape[2]
        mean = grouped.sum(dim=(0, 2), dtype=torch.float32) / count
        var = torch.linalg.vector_norm(grouped, dim=(0, 2), dtype=torch.float32).square() / count - mean.square()
        scale = torch.rsqrt(var + self.eps)[:, None] * comfy.ops.cast_to(self.weight, torch.float32, x.device).view(self.groups, -1)
        shift = comfy.ops.cast_to(self.bias, torch.float32, x.device).view(self.groups, -1) - mean[:, None] * scale
        return scale.flatten().to(x.dtype), shift.flatten().to(x.dtype)

    def forward(self, x):
        scale, shift = self.affine(x)
        return F.silu(torch.addcmul(shift, x, scale, out=x), inplace=True)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = PointGroupNormSiLU(in_channels)
        self.conv1 = SparseConv3d(in_channels, out_channels)
        self.norm2 = PointGroupNormSiLU(out_channels)
        self.conv2 = SparseConv3d(out_channels, out_channels)
        self.skip = ops.Linear(in_channels, out_channels) if in_channels != out_channels else None

    def forward(self, x):
        h = self.conv1(x, norm=self.norm1)
        return self.conv2(h.replace(self.norm2(h.feats)), out=x.feats if self.skip is None else self.skip(x.feats))


def z_order(coords):
    """Argsort of the Morton code of [N, 3] grid coordinates."""
    depth = int(coords.max()).bit_length()
    coords = coords.long()
    code = torch.zeros_like(coords[:, 0])
    for bit in range(depth):
        code |= ((coords[:, 0] >> bit) & 1) << (3 * bit + 2)
        code |= ((coords[:, 1] >> bit) & 1) << (3 * bit + 1)
        code |= ((coords[:, 2] >> bit) & 1) << (3 * bit)
    return torch.argsort(code)


class AttentionBlock(nn.Module):
    """Windowed self-attention along a z-order curve, with a sparse-conv positional encoding."""
    def __init__(self, channels, num_heads, transpose, window=48, mlp_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.transpose = transpose
        self.window = window
        self.cpe_conv = SparseConv3d(channels, channels)
        self.cpe_linear = ops.Linear(channels, channels)
        self.cpe_norm = ops.LayerNorm(channels)
        self.norm1 = ops.LayerNorm(channels)
        self.qkv = ops.Linear(channels, channels * 3)
        self.proj = ops.Linear(channels, channels)
        self.norm2 = ops.LayerNorm(channels)
        self.fc1 = ops.Linear(channels, channels * mlp_ratio)
        self.fc2 = ops.Linear(channels * mlp_ratio, channels)
        self.comfy_attention = ComfyAttention()

    def forward(self, x):
        feats = x.feats + self.cpe_norm(self.cpe_linear(self.cpe_conv(x).feats))
        grid = x.coords[:, 1:]
        order = z_order(grid[:, [1, 0, 2]] if self.transpose else grid)
        n, c = feats.shape
        gather, window = order, n
        if n > self.window:
            window = self.window
            n_pad = (n + window - 1) // window * window
            if n_pad != n:
                # pad the last window by repeating the tail of the previous one
                pad = torch.arange(n_pad, device=feats.device)
                pad[n:] = pad[n - window:n_pad - window].clone()
                gather = order[pad]
        qkv = self.qkv(self.norm1(feats))[gather].view(-1, window, 3, self.num_heads, c // self.num_heads)
        q, k, v = qkv.unbind(2)
        out = dense_attention(q, k, v, preferred_attention=self.comfy_attention, low_precision_attention=False).reshape(-1, c)[:n]
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(n, device=order.device)
        feats = feats + self.proj(out[inverse])
        feats = feats + self.fc2(F.gelu(self.fc1(self.norm2(feats))))
        return x.replace(feats)


class Unpool(nn.Module):
    """Split every voxel into its eight children and keep the ones predicted occupied."""
    def __init__(self, channels):
        super().__init__()
        self.cls_norm = PointGroupNormSiLU(channels)
        self.cls_conv = SparseConv3d(channels, 8)
        self.proj = ops.Linear(channels, channels * 8, bias=False)

    def forward(self, x, resolution):
        occupied = self.cls_conv(x, norm=self.cls_norm).feats > 0
        counts = occupied.sum(0).tolist()
        total = sum(counts)
        if total == 0:
            raise RuntimeError("Hi3D VAE Unpool predicted 0 occupied voxels (non-finite or degenerate latent features).")
        c = x.feats.shape[1]
        feats = torch.empty(total, c, device=x.feats.device, dtype=x.feats.dtype)
        coords = torch.empty(total, 4, device=x.coords.device, dtype=x.coords.dtype)
        start = 0
        for child, count in enumerate(counts):
            parents = occupied[:, child]
            weight = comfy.ops.cast_to_input(self.proj.weight[child * c:(child + 1) * c], x.feats)
            torch.mm(x.feats[parents], weight.t(), out=feats[start:start + count])
            offset = torch.tensor([0, (child >> 2) & 1, (child >> 1) & 1, child & 1], dtype=x.coords.dtype, device=x.coords.device)
            coords[start:start + count] = x.coords[parents] * torch.tensor([1, 2, 2, 2], dtype=x.coords.dtype, device=x.coords.device) + offset
            start += count
        return sparse_tensor(feats, coords, resolution)


class DecoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, upsample):
        super().__init__()
        self.blocks = nn.ModuleList([ResBlock(in_channels, out_channels), ResBlock(out_channels, out_channels)])
        self.up = Unpool(out_channels) if upsample else None


class ConvOut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = PointGroupNormSiLU(in_channels)
        self.conv = SparseConv3d(in_channels, out_channels)

    def forward(self, x):
        return self.conv(x.replace(self.norm(x.feats))).feats


def corner_field(out, coords, resolution):
    """Average the decoder's per-cube corner sign logits, distances and offsets ([N, 40]) onto the unique grid
    vertices. Returns (vertices, signed distances, [N, 8] corner indices, cube coords) with cubes sorted by grid key."""
    order = torch.argsort((coords[:, 0].long() * resolution + coords[:, 1]) * resolution + coords[:, 2])
    coords = coords[order]
    side = resolution + 1
    corners = torch.tensor([(x * side + y) * side + z for x, y, z in CUBE_CORNERS], device=coords.device)
    keys = ((coords[:, 0].long() * side + coords[:, 1]) * side + coords[:, 2])[:, None] + corners
    keys, cubes = torch.unique(keys.flatten(), return_inverse=True)
    verts = torch.stack((keys // side ** 2, keys // side % side, keys % side), dim=1).float()
    del keys
    cubes = cubes.view(-1, 8).int()
    # chunked so the [N, 8, 5] float32 corner attributes are never materialized
    merged = torch.zeros((verts.shape[0], 5), device=verts.device)
    for start in range(0, cubes.shape[0], 1 << 20):
        rows = out[order[start:start + (1 << 20)]].float()
        feats = torch.cat((torch.sigmoid(rows[:, :8]).unsqueeze(2), torch.tanh(rows[:, 8:16]).unsqueeze(2), torch.tanh(rows[:, 16:]).view(-1, 8, 3) * 0.5), dim=2)
        merged.index_add_(0, cubes[start:start + (1 << 20)].flatten(), feats.flatten(0, 1))
    merged /= torch.bincount(cubes.flatten(), minlength=verts.shape[0])[:, None]
    sdf = torch.where(merged[:, :1] >= 0.5, 1.0, -1.0) * merged[:, 1:2].abs()
    return verts.add_(merged[:, 2:5]), sdf, cubes, coords


def _boundary_loops(boundary):
    outgoing = {}
    for idx, (start, _) in enumerate(boundary):
        outgoing.setdefault(int(start), []).append(idx)
    visited = [False] * len(boundary)
    loops = []
    for i in range(len(boundary)):
        if visited[i]:
            continue
        visited[i] = True
        start = int(boundary[i][0])
        loop = [start, int(boundary[i][1])]
        while loop[-1] != start:
            for next_idx in outgoing.get(loop[-1], []):
                if not visited[next_idx]:
                    visited[next_idx] = True
                    loop.append(int(boundary[next_idx][1]))
                    break
            else:
                break
        if loop[0] == loop[-1] and len(loop) > 3:
            loop.pop()
            loops.append(loop)
    return loops


def _ear_clip(points, indices):
    faces = []
    while True:
        if len(indices) == 3:
            faces.append(indices[[2, 1, 0]])
            break
        next_pts = np.roll(points, -1, axis=0)
        tangent = next_pts - points
        tangent = tangent / (np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-6)
        prev = np.roll(tangent, 1, axis=0)
        normal = np.cross(-prev, tangent)
        center = points.mean(axis=0, keepdims=True)
        sin_sign = np.sign((np.cross(next_pts - center, points - center) * normal).sum(axis=1, keepdims=True))
        sin_val = np.linalg.norm(normal, axis=1, keepdims=True) * sin_sign
        cos_val = -(prev * tangent).sum(axis=1, keepdims=True)
        angle = np.arctan2(sin_val, cos_val).reshape(-1)
        collapsed = angle == 0.0
        ear = int(np.argmax(np.where(collapsed, np.inf, 1.0 / np.where(collapsed, 1.0, angle))))
        faces.append(indices[[(ear + 1) % len(indices), ear, (ear - 1) % len(indices)]])
        points = np.delete(points, ear, axis=0)
        indices = np.delete(indices, ear)
    return np.vstack(faces)


def fill_holes(vertices, faces, max_loop=10000):
    """Close boundary loops by ear clipping."""
    directed = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    _, inverse, counts = torch.unique(directed.min(dim=1).values.long() * vertices.shape[0] + directed.max(dim=1).values, return_inverse=True, return_counts=True)
    boundary = directed[counts[inverse] == 1].cpu().numpy()
    loops = [loop for loop in _boundary_loops(boundary) if len(loop) <= max_loop]
    if not loops:
        return faces
    points = vertices.cpu().numpy()
    new_faces = np.vstack([_ear_clip(points[loop], np.array(loop)) for loop in loops])
    return torch.cat((faces, torch.from_numpy(new_faces).to(faces)))


class Hi3DVae(nn.Module):
    def __init__(self, z_channels=32, channels=(128, 256, 512, 512), mid_heads=32, out_channels=32):
        super().__init__()
        mid = channels[-1]
        self.z_to_f = SparseConv3d(z_channels, mid)
        self.mid = nn.ModuleList([ResBlock(mid, mid), AttentionBlock(mid, mid_heads, False), AttentionBlock(mid, mid_heads, True), ResBlock(mid, mid)])
        stages = []
        for s in reversed(range(len(channels))):
            stages.append(DecoderStage(channels[s], channels[max(s - 1, 0)], upsample=s > 0))
        self.dec = nn.ModuleList(stages)
        self.conv_out = ConvOut(channels[0], 8 + out_channels)

    def decode(self, latent, coords, resolution):
        """latent [N, C] on the [N, 3] coords of a resolution^3 grid -> (vertices in grid units, faces)."""
        coords = torch.cat((torch.zeros_like(coords[:, :1]), coords), dim=1).int()
        x = self.z_to_f(sparse_tensor(latent, coords, resolution))
        for block in self.mid:
            x = block(x)
        for stage in self.dec:
            for block in stage.blocks:
                x = block(x)
            if stage.up is not None:
                resolution *= 2
                x = stage.up(x, resolution)
        out = self.conv_out(x)
        coords = x.coords[:, 1:]
        del x  # each extraction step frees what it no longer needs; at 1536 the surface barely fits in 6 GB
        verts, sdf, cubes, coords = corner_field(out, coords, resolution)
        del out
        vertices, faces = FlexiCubes(device=verts.device)(verts, sdf, cubes, resolution, coords)
        del verts, sdf, cubes, coords
        return vertices, fill_holes(vertices, faces)
