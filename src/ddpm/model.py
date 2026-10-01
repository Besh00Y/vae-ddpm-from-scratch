import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    """Turns the timestep t into a vector, same idea as positional encoding in transformers."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]  # inject time
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Single-head self-attention over all spatial positions."""

    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)
        attn = torch.softmax(torch.einsum("bci,bcj->bij", q, k) / math.sqrt(c), dim=-1)
        out = torch.einsum("bij,bcj->bci", attn, v).reshape(b, c, h, w)
        return x + self.proj(out)


class Block(nn.Module):
    """ResBlock + optional attention."""

    def __init__(self, in_ch, out_ch, time_dim, use_attn, dropout):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, time_dim, dropout)
        self.attn = AttentionBlock(out_ch) if use_attn else nn.Identity()

    def forward(self, x, t_emb):
        return self.attn(self.res(x, t_emb))


class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb=None):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x, t_emb=None):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(self, in_channels=3, base=64, ch_mult=(1, 2, 2, 2), num_res=2,
                 attn_resolutions=(16,), image_size=32, dropout=0.1):
        super().__init__()
        time_dim = base * 4
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(base),
            nn.Linear(base, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.in_conv = nn.Conv2d(in_channels, base, 3, padding=1)

        # ---- encoder (down path) ----
        self.down = nn.ModuleList()
        chs, ch, res = [base], base, image_size
        for i, mult in enumerate(ch_mult):
            out = base * mult
            for _ in range(num_res):
                self.down.append(Block(ch, out, time_dim, res in attn_resolutions, dropout))
                ch = out
                chs.append(ch)
            if i != len(ch_mult) - 1:
                self.down.append(Down(ch))
                chs.append(ch)
                res //= 2

        # ---- bottleneck ----
        self.mid1 = Block(ch, ch, time_dim, True, dropout)
        self.mid2 = Block(ch, ch, time_dim, False, dropout)

        # ---- decoder (up path), consumes the skip connections ----
        self.up = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out = base * mult
            for _ in range(num_res + 1):
                self.up.append(Block(ch + chs.pop(), out, time_dim, res in attn_resolutions, dropout))
                ch = out
            if i != 0:
                self.up.append(Up(ch))
                res *= 2

        self.out = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, in_channels, 3, padding=1)
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = self.in_conv(x)
        hs = [h]
        for layer in self.down:
            h = layer(h, t_emb)
            hs.append(h)

        h = self.mid2(self.mid1(h, t_emb), t_emb)

        for layer in self.up:
            if isinstance(layer, Block):
                h = torch.cat([h, hs.pop()], dim=1)  # skip connection
            h = layer(h, t_emb)
        return self.out(h)