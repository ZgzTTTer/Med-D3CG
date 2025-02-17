import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        return self.timembedding(t)

class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()
    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)
    def forward(self, x, temb):
        return self.main(x)

class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()
    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)
    def forward(self, x, temb):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.main(x)

class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1)
        self.proj = nn.Conv2d(in_ch, in_ch, 1)
        self.initialize()
        self.attention_map = None

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self,x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.reshape(B, C, H*W).permute(0,2,1)
        k = k.reshape(B, C, H*W)
        w = torch.bmm(q, k)*(C**(-0.5))
        w = F.softmax(w, dim=-1)
        self.attention_map = w

        v = v.reshape(B, C, H*W).permute(0,2,1)
        h = torch.bmm(w,v)
        h = h.permute(0,2,1).reshape(B,C,H,W)
        h = self.proj(h)
        return x+h

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32,in_ch),
            Swish(),
            nn.Conv2d(in_ch,out_ch,3,1,1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim,out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32,out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch,out_ch,3,1,1),
        )
        self.shortcut = nn.Conv2d(in_ch,out_ch,1) if in_ch!=out_ch else nn.Identity()
        self.attn = AttnBlock(out_ch) if attn else nn.Identity()
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)
        init.xavier_uniform_(self.block2[-1].weight,gain=1e-5)

    def forward(self,x,temb):
        h=self.block1(x)
        h+=self.temb_proj(temb)[:,:,None,None]
        h=self.block2(h)
        h=h+self.shortcut(x)
        h=self.attn(h)
        return h