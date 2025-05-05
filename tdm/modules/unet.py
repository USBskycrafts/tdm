import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.SiLU(),
            nn.Linear(dim*4, dim)
        )
        
    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.mlp(emb)
    
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, cond_dim, num_contrasts, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(tdim, out_ch))
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.cond_proj = nn.Conv2d(cond_dim, out_ch, 1) if cond_dim > 0 else None
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.contrast_embed = nn.Parameter(torch.randn(num_contrasts, out_ch, 1, 1)) if num_contrasts > 0 else None

    def forward(self, x, t, aim=None, cond=None):
        h = self.block1(x)
        h += self.mlp(t)[..., None, None]
        if cond is not None and self.cond_proj is not None:
            cond = self.cond_proj(cond)
            
            # a brute force way to fuse the features
            # h += F.interpolate(cond, size=h.shape[2:], mode='bilinear', align_corners=False)
            # or a more ellegant way
            score = torch.einsum('b c h w, b c u v -> b h w u v', h, cond)
            score = torch.softmax(score / torch.sqrt(torch.tensor(h.shape[1])), dim=1)
            score = torch.einsum('b h w u v, b c u v -> b c h w', score, cond)
            h = h + score
        if aim is not None and self.contrast_embed is not None:
            h += self.contrast_embed[aim]
        h = self.block2(h)
        return h + self.res_conv(x)
    
    
class Attention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.norm = nn.GroupNorm(32, dim)
        self.qkv = nn.Conv2d(dim, dim*3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        attn = torch.einsum('b h c i, b h c j -> b h i j', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('b h i j, b h c j -> b h i c', attn, v)
        out = rearrange(out, 'b h (x y) c -> b (h c) x y', x=h, y=w)
        return self.proj(out) + x


class UNet(nn.Module):
    def __init__(
        self,
        in_ch=3,
        base_ch=64,
        ch_scales=[1,2,4,8],
        num_res_blocks=2,
        num_contrasts=4,
        attn_scales=[False, False, True, False],
        cond_dim=512,
        dropout=0.1
    ):
        super().__init__()
        self.tdim = base_ch * 4
        self.temb = TimestepEmbedding(self.tdim)
        
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.in_ch = in_ch
        
        # Input conv
        self.in_conv = nn.Sequential(
            nn.Conv2d(self.in_ch, base_ch, 3, padding=1),
            nn.GroupNorm(32, base_ch),
            nn.SiLU(inplace=True)
        )
        
        # Downsample
        in_channels = [base_ch * s for s in ch_scales[:-1]]
        out_channels = [base_ch * s for s in ch_scales[1:]]
        for i, (in_ch, out_ch) in enumerate(zip(in_channels, out_channels)):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(in_ch, out_ch, self.tdim, cond_dim, num_contrasts, dropout))
                if attn_scales[i]:
                    blocks.append(Attention(out_ch))
                in_ch = out_ch
            self.down_blocks.append(blocks)
            if i < len(ch_scales)-1:
                self.down_blocks.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
        
        # Middle
        mid_ch = base_ch * ch_scales[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, self.tdim, cond_dim, num_contrasts, dropout)
        self.mid_attn = Attention(mid_ch)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, self.tdim, cond_dim, num_contrasts, dropout)
        
        # Upsample
        in_channels = [base_ch * s for s in reversed(ch_scales)]
        out_channels = [base_ch * s for s in reversed(ch_scales[:-1])]
        for i, (in_ch, out_ch) in enumerate(zip(in_channels, out_channels)):
            blocks = nn.ModuleList()
            blocks.append(nn.Conv2d(in_ch * 2, in_ch, 3, padding=1))
            for _ in range(num_res_blocks+1):
                blocks.append(ResBlock(in_ch, out_ch, self.tdim, cond_dim, num_contrasts, dropout))
                if attn_scales[::-1][i]:
                    blocks.append(Attention(out_ch))
                in_ch = out_ch
            self.up_blocks.append(blocks)
            if i < len(ch_scales)-1:
                self.up_blocks.append(nn.Upsample(scale_factor=2))
        
        self.out_conv = nn.Sequential(
            nn.GroupNorm(32, base_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, self.in_ch, 3, padding=1),
        )

    def forward(self, x, t, aim, cond_features):
        t = self.temb(t)
        
            
        # Input conv
        x = self.in_conv(x)
        
        # Downsample
        hs = [x]
        cond_idx = 0
        for block in self.down_blocks:
            if isinstance(block, nn.ModuleList):
                for b in block:
                    if isinstance(b, ResBlock):
                        x = b(x, t, aim, cond_features)
                    else:
                        x = b(x)
                cond_idx += 1
            else:
                x = block(x)
                hs.append(x)
        # Middle
        x = self.mid_block1(x, t, aim, cond_features)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, aim, cond_features)
        
        
        # Upsample
        for block in self.up_blocks:
            if isinstance(block, nn.ModuleList):
                x = torch.cat([x, hs.pop()], dim=1)
                for b in block:
                    if isinstance(b, ResBlock):
                        x = b(x, t, aim, cond_features)
                    else:
                        x = b(x)
                cond_idx += 1
            else:
                x = block(x)
        return self.out_conv(x)
