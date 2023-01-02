from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from functools import partial
from einops import rearrange

from timm.models.layers import DropPath


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TranslationClassifier(torch.nn.Module):
    def __init__(
        self,
        in_dim=18,
        embed_dim=16,
        num_layers=4,
        dropout=0.25,
        num_classes=4
    ) -> None:
        super(TranslationClassifier, self).__init__()
        
        self.word2vec_fc1 = nn.Linear(in_dim, embed_dim)
        self.word2vec_fc2 = nn.Linear(embed_dim, embed_dim)
        
        self.temporal_embed = nn.Parameter(torch.zeros(1, 256, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, 0.1, num_layers)]
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=4, mlp_ratio=2, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(num_layers)])
        self.temporal_norm = norm_layer(embed_dim)
        self.weighted_mean = torch.nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1)
        
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
        self.drop = nn.Dropout(dropout)
        
        # activation functions
        self.act_layer = nn.ReLU()
    
    
    def forward_features(self, x):
        b  = x.shape[0] # (B, L, embed)
        x += self.temporal_embed
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.temporal_norm(x)
        ##### x.shape = (B, L, embed)
        x = self.weighted_mean(x)
        x = x.view(b, -1)
        return x

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        x = batch['input'] # (B, L, 18)
        y = batch['gt']
        
        x = self.drop(self.act_layer(self.word2vec_fc1(x)))
        x = self.drop(self.act_layer(self.word2vec_fc2(x)))
        
        x = self.forward_features(x)
        x = self.head(x)
        
        return x.max(1)[1], F.cross_entropy(x, y.long())

if __name__ == '__main__':
    test_layer = TranslationClassifier(embed_dim=256)
    
    model_params = 0
    for parameter in test_layer.parameters():
        model_params += parameter.numel()
    print('Trainable parameter count:', model_params)
    
    # A = torch.randn(64,256,18)
    # B = torch.ones(64)
    # test_layer({'input': A, 'gt': B})