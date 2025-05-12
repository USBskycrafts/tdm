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


# 更新 ResBlock 使用统一 Attention
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, cond_dim, num_contrasts, dropout=0.1, attn_heads=4):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(tdim, out_ch))
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch), nn.SiLU(), nn.Conv2d(
                in_ch, out_ch, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch), nn.SiLU(), nn.Dropout(
                dropout), nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.res_conv = nn.Conv2d(
            in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.cond_proj = nn.Conv2d(
            cond_dim, out_ch, 1) if cond_dim > 0 else None
        self.contrast_embed = nn.Parameter(torch.randn(
            num_contrasts, out_ch, 1, 1)) if (num_contrasts > 0 and self.cond_proj is not None) else None
        # 支持自注意和交叉注意
        self.attn = Attention(out_ch, heads=attn_heads)

    def forward(self, x, t, aim=None, cond=None):
        h = self.block1(x)
        h += self.mlp(t)[..., None, None]
        if cond is not None and self.cond_proj is not None:
            cond_feat = self.cond_proj(cond)
            # 交叉注意力融合
            h = h + self.attn(h, cond_feat)
        if aim is not None and self.contrast_embed is not None:
            h = h + self.contrast_embed[aim]
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, use_flash=True):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.use_flash = use_flash
        # Single proj for QKV
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(32, dim)

    def forward(self, x, context=None):
        """
        x: [B, C, H, W] target tensor
        context: [B, C, H, W] source tensor for cross-attention; if None, do self-attention
        """
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        if context is None:
            # self-attention: context = x
            context = x_norm
        else:
            # cross-attention: normalize context separately
            context = self.norm(context)
        # compute Q from x_norm, K/V from context
        q = self.qkv(x_norm)[:, :c]
        kv = self.qkv(context)
        k, v = torch.chunk(kv[:, c:], 2, dim=1)
        # reshape for multi-head
        q = rearrange(q, 'b (h d) H W -> b h (H W) d', h=self.heads)
        k = rearrange(k, 'b (h d) H W -> b h (H W) d', h=self.heads)
        v = rearrange(v, 'b (h d) H W -> b h (H W) d', h=self.heads)
        # scaled dot-product (FlashAttention if available)
        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scale,
                enable_gqa=False,
            )
        else:
            attn = torch.einsum(
                'b h i d, b h j d -> b h i j', q, k) * self.scale
            attn = attn.softmax(dim=-1)
            attn_out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        # reshape back and project
        attn_out = rearrange(attn_out, 'b h (H W) d -> b (h d) H W', H=h, W=w)
        return self.proj(attn_out) + x


class UNet(nn.Module):
    def __init__(
        self,
        in_ch=3,
        base_ch=64,
        ch_scales=[1, 2, 4, 8],
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
            for j in range(num_res_blocks):
                current_cond_dim = cond_dim if j == 0 else 0
                blocks.append(ResBlock(in_ch, out_ch, self.tdim,
                              current_cond_dim, num_contrasts, dropout))
                if attn_scales[i]:
                    blocks.append(Attention(out_ch))
                in_ch = out_ch
            self.down_blocks.append(blocks)
            if i < len(ch_scales)-1:
                self.down_blocks.append(
                    nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))

        # Middle
        mid_ch = base_ch * ch_scales[-1]
        self.mid_block1 = ResBlock(
            mid_ch, mid_ch, self.tdim, cond_dim, num_contrasts, dropout)
        self.mid_attn = Attention(mid_ch)
        self.mid_block2 = ResBlock(
            mid_ch, mid_ch, self.tdim, cond_dim, num_contrasts, dropout)

        # Upsample
        in_channels = [base_ch * s for s in reversed(ch_scales)]
        out_channels = [base_ch * s for s in reversed(ch_scales[:-1])]
        for i, (in_ch, out_ch) in enumerate(zip(in_channels, out_channels)):
            blocks = nn.ModuleList()
            blocks.append(nn.Conv2d(in_ch * 2, in_ch, 3, padding=1))
            for j in range(num_res_blocks+1):
                current_cond_dim = cond_dim if j == 0 else 0
                blocks.append(ResBlock(in_ch, out_ch, self.tdim,
                              current_cond_dim, num_contrasts, dropout))
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
