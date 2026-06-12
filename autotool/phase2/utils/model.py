# ------------------------------------------------------------
# File: model.py
# ------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
#
from phase2_train.utils.data_loader import exemplar_norm_function

###
def custom_vit_init(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


###############################################################################
# Sine-Cosine 1D Positional Embedding (from original SimpleViT approach)
###############################################################################
def posemb_sincos_1d(length, dim, temperature=10000, dtype=torch.float32):

    assert dim % 2 == 0, "feature dimension must be even for 1D sin-cos embedding."
    position = torch.arange(length, dtype=dtype).unsqueeze(1)
    
    omega = torch.arange(dim // 2, dtype=dtype) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)
    angles = position * omega.unsqueeze(0)
    emb = torch.cat([angles.sin(), angles.cos()], dim=1)
    return emb


###############################################################################
# Sine-Cosine 2D Positional Embedding (from original SimpleViT approach)
###############################################################################
def posemb_sincos_2d(h, w, dim, temperature=10000, dtype=torch.float32):

    y, x = torch.meshgrid(
        torch.arange(h, dtype=dtype),
        torch.arange(w, dtype=dtype),
        indexing='ij'
    )
    assert dim % 4 == 0, "feature dimension must be multiple of 4 for sin-cos embedding."

    omega = torch.arange(dim // 4, dtype=dtype) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    pe = torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)

    return pe


###############################################################################
# Sine-Cosine 3D Positional Embedding (from original SimpleViT approach)
###############################################################################
def posemb_sincos_3d(patches, temperature = 10000, dtype = torch.float32):
    _, f, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    assert dim % 6 == 0, "feature dimension must be multiple of 6 for sin-cos embedding."

    z, y, x = torch.meshgrid(
        torch.arange(f, device = device),
        torch.arange(h, device = device),
        torch.arange(w, device = device),
    indexing = 'ij')

    fourier_dim = dim // 6

    omega = torch.arange(fourier_dim, device = device) / (fourier_dim - 1)
    omega = 1. / (temperature ** omega)

    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim = 1)

    pe = F.pad(pe, (0, dim - (fourier_dim * 6)))
    
    return pe.type(dtype)


###############################################################################
# FeedForward
###############################################################################

class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout_prob=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_prob) if dropout_prob > 0 else nn.Identity(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_prob) if dropout_prob > 0 else nn.Identity(),
        )
    def forward(self, x):
        return self.net(x)


###############################################################################
# Multi-head Self-Attention
###############################################################################
class Attention(nn.Module):
  
    def __init__(self, dim, heads, dim_head, num_patches, dropout_prob=-1):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.num_patches = num_patches
        self.scale = dim_head ** -0.5

        if dropout_prob > 0:
            self.norm = nn.LayerNorm(dim)
            self.attend = nn.Sequential(
                nn.Softmax(dim=-1),
                nn.Dropout(p=dropout_prob)
            )
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim, bias=False),
                nn.Dropout(p=dropout_prob)
            )
        else:
            self.norm = nn.LayerNorm(dim)
            self.attend = nn.Softmax(dim=-1)
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
            self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, causal_flag=True):

        b, n, d = x.shape
        assert n % self.num_patches == 0
        seq_len = n // self.num_patches
        
        #
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        ###
        if causal_flag:
            mask = torch.tril(torch.ones(n, n, device=x.device, dtype=torch.bool))
            mask = mask.unsqueeze(0).unsqueeze(0)
            dots = dots.masked_fill(~mask, float('-inf'))

        ###
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


###############################################################################
# TransformerBlock
###############################################################################
class TransformerBlock(nn.Module):

    def __init__(self, dim, heads, mlp_dim, num_patches, dim_head=64, dropout_prob=-1, causal_flag=True):
        super().__init__()
        self.causal_flag = causal_flag
        self.attn = Attention(dim=dim, heads=heads, dim_head=dim_head, num_patches=num_patches, 
                              dropout_prob=dropout_prob)
        self.ff = FeedForward(dim, mlp_dim, dropout_prob=dropout_prob)

    def forward(self, x):
        x = x + self.attn(x, causal_flag=self.causal_flag)
        x = x + self.ff(x)
        
        return x


###############################################################################
# SequenceViT
###############################################################################
class SequenceViT(nn.Module):

    def __init__(
        self,
        args,
        image_size,    # (H, W)
        patch_size,    # (pH, pW)
        seq_len,
        horizon,
        dim,
        depth,
        heads,
        mlp_dim,
        init_query_emb=None,
        dim_head=64,
        device='cuda',
        pos_emb_pattern='3d',
        pe_temperature=10000,
        te_intensity=0.5,
        causal_flag=True
    ):
        
        super().__init__()
        self.img_h, self.img_w = image_size
        self.p_h, self.p_w = patch_size
        assert self.img_h % self.p_h == 0, "image height not divisible"
        assert self.img_w % self.p_w == 0, "image width not divisible"
        self.num_patches = (self.img_h // self.p_h) * (self.img_w // self.p_w)
        patch_dim = self.p_h * self.p_w

        self.seq_len = seq_len
        self.dim = dim
        self.device = device
        self.pos_emb_pattern = pos_emb_pattern
        self.pe_temperature = pe_temperature
        self.causal_flag = causal_flag
        self.dropout_prob = args.dropout_prob

        if pos_emb_pattern == '3d':
            self.to_patch_embedding_3d = nn.Sequential(
                Rearrange('b c (f pf) (h p1) (w p2) -> b f h w (p1 p2 pf c)', p1 = self.p_h, p2 = self.p_w, pf = 1),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, dim),
                nn.LayerNorm(dim),
            )
        else:
            self.patch_embed = nn.Sequential(
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, dim),
                nn.LayerNorm(dim),
            )

            ph_count = self.img_h // self.p_h
            pw_count = self.img_w // self.p_w
            #
            self.pos_embedding = posemb_sincos_2d(
                h=ph_count,
                w=pw_count,
                dim=dim,
                temperature=pe_temperature
            ).to(self.device)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, heads=heads, mlp_dim=mlp_dim, num_patches=self.num_patches, dim_head=dim_head, 
                             dropout_prob=self.dropout_prob,
                             causal_flag=causal_flag)
            for _ in range(depth)
        ])

        self.final_header = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim),
        )

        self.query_param = nn.Parameter(torch.randn(self.img_h, self.img_w), requires_grad=True)
        
        if init_query_emb is not None:
            randomized_emb_m = init_query_emb + 0.01 * torch.randn_like(init_query_emb)
            self.query_param.data = exemplar_norm_function(randomized_emb_m.unsqueeze(0)).squeeze(0)
        else:
            randomized_emb_m = 0.01 * torch.randn_like(self.query_param.data)
            self.query_param.data = exemplar_norm_function(randomized_emb_m.unsqueeze(0)).squeeze(0)

        ###################
        self.query_param.requires_grad = False
        self.temporal_embedding = nn.Embedding(horizon + 5, dim)
        ###
        nn.init.uniform_(self.temporal_embedding.weight, -te_intensity, te_intensity)

    def forward(self, seq_images):

        b, s, h, w = seq_images.shape
        
        hp = h // self.p_h
        wp = w // self.p_w
        
        ##############################################################################
        if self.pos_emb_pattern == '3d':
            seq_images = seq_images.unsqueeze(1)
            x = self.to_patch_embedding_3d(seq_images)
            #
            pe = posemb_sincos_3d(x, temperature=self.pe_temperature)
            #
            x = rearrange(x, 'b ... d -> b (...) d') + pe
        elif self.pos_emb_pattern == '2d_sequential':
            #####
            x = rearrange(
                seq_images,
                'b s (hp ph) (wp pw) -> b s (hp wp) (ph pw)',
                ph=self.p_h, pw=self.p_w, hp=hp, wp=wp
            )

            b_, s_, np_, pd_ = x.shape
            x = x.reshape(b * s, np_, pd_)
            x = self.patch_embed(x)  # => (b*s, num_patches, dim)
            x = x.reshape(b, s, np_, self.dim)
            
            spatial_pe = self.pos_embedding.unsqueeze(0).unsqueeze(1)  # => (1, 1, num_patches, dim)
            temporal_pe = posemb_sincos_1d(s, self.dim, temperature=self.pe_temperature, dtype=x.dtype).to(self.device)  # => (s, dim)
            temporal_pe = temporal_pe.unsqueeze(0).unsqueeze(2)  # => (1, s, 1, dim)
            
            x = x + spatial_pe + temporal_pe
            x = x.reshape(b, s * self.num_patches, self.dim)
        else:
            raise NotImplementedError
        

        ##############################################################################
        temporal_indices = torch.arange(s, device=x.device)               \
                    .unsqueeze(1)                                           \
                    .expand(s, self.num_patches)                     \
                    .contiguous()                                           \
                    .view(-1)

        temp_emb = self.temporal_embedding(temporal_indices)
        temp_emb = temp_emb.unsqueeze(0).expand(b, -1, -1)
        x = x + temp_emb

        ##############################################################################
        for blk in self.blocks:
            x = blk(x)

        x = self.final_header(x)
        x = x.reshape(b, s, self.num_patches, self.p_h*self.p_w)
        x = x.reshape(b, s, hp, wp, self.p_h, self.p_w)
        out = rearrange(
            x,
            'b s hp wp ph pw -> b s (hp ph) (wp pw)'
        ) 

        return out
