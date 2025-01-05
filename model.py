import math
from typing import NamedTuple
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

def fit_tensor(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        while len(x.shape) < len(target.shape):
            x = x[None]
        return x


class MultiHeadSelfAttention(nn.Module):
    qkv: nn.Linear
    out: nn.Linear
    scale: float

    def __init__(
        self, 
        input_dim: int,
        heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.qkv = nn.Linear(input_dim, input_dim * 3)
        self.out = nn.Linear(input_dim, input_dim)
        self.scale = math.sqrt(input_dim / heads)
        self.heads = heads
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        x: torch.Tensor,
    ) -> torch.Tensor:
        dim = x.size(-1) // self.heads
        qkv: torch.Tensor = self.qkv(x)
        qkv = einops.rearrange(qkv, 'b n (c h d) -> b c h n d', h=self.heads, d=dim, c=3)
        q, k, v = map(lambda x: x.squeeze(1), qkv.chunk(3, dim=1))
        q = q / self.scale
        k = k.transpose(-1, -2)
        attn = q @ k
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        context = attn @ v
        context = einops.rearrange(context, 'b h n d -> b n (h d)')
        out = self.out(context)
        return out

class AttnBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(input_dim=input_dim, heads=heads)
        self.norm = nn.GroupNorm(8, num_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm(x)
        out = self.attn(out)
        return self.dropout(out) + x
        
class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, num_channels)
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        x: torch.Tensor,
    ) -> torch.Tensor:
        out = self.norm(x)
        out = self.dropout(F.relu(self.fc1(out))) * self.fc2(out)
        return self.dropout(out) + x

class TransformerBlock(nn.Module):
    attnBlock: AttnBlock
    residual_block: ResidualBlock

    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.attnBlock = AttnBlock(input_dim, num_channels, num_heads)
        self.residual_block = ResidualBlock(input_dim, num_channels)
    def forward(
        self, 
        x: torch.Tensor,
    ) -> torch.Tensor:
        out = self.attnBlock(x)
        out = self.residual_block(out)
        return out

# class Classifier(nn.Module):
#     def __init__(self, input_dim: int) -> None:
#         super().__init__()
#         self.input_fc = nn.Linear(input_dim, 512)
#         self.seq = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#         )
#         self.output_fc = nn.Linear(512, 1)
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.unsqueeze(1)
#         x = self.input_fc(x)
#         x = self.seq(x)
#         x = F.relu(x)
#         x = self.output_fc(x.flatten(1))
#         return x.sigmoid()

class Classifier(nn.Module):

    def __init__(
        self, 
        input_dim: int, 
    ) -> None:
        super().__init__()
        self.classify_token = nn.Parameter(torch.randn(1, 512))
        self.norm = nn.BatchNorm1d(input_dim)
        self.input_fc = nn.Linear(input_dim, 512) 
        self.input_conv = nn.Conv1d(1, 64 - 1, 1)
        self.blocks = nn.ModuleList([
            TransformerBlock(512, 64, 4),
            TransformerBlock(512, 64, 4),
            TransformerBlock(512, 64, 4),
            TransformerBlock(512, 64, 4),
            TransformerBlock(512, 64, 4),
        ])
        self.out = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = x.unsqueeze(1)
        x = self.input_fc(x)
        x = self.input_conv(x)
        x = torch.cat([self.classify_token.expand(x.shape[0], 1, 512), x], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.out.forward(x[:, 0]).sigmoid()
