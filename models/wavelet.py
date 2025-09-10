import torch
from torch import nn
from torch.nn import init
from .blocks import Swish, TimeEmbedding, DownSample, UpSample, ResBlock

class WTUNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout, in_channels=None, out_channels=None):
        super().__init__()
        tdim = ch*4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.out_channels = out_channels

        wavelet_in_channels = 24 if out_channels == 3 else 8

        self.head = nn.Conv2d(wavelet_in_channels, ch, 3, 1, 1)
        self.initialize(self.head)

        now_ch = ch
        chs = [ch]
        self.downblocks = nn.ModuleList()
        for i, mult in enumerate(ch_mult):
            out_ch = ch*mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(now_ch, out_ch, tdim, dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult)-1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch*mult
            for _ in range(num_res_blocks+1):
                self.upblocks.append(ResBlock(chs.pop()+now_ch, out_ch, tdim, dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))

        wavelet_out_channels = 12 if out_channels == 3 else 4

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, wavelet_out_channels, 3, 1, 1)
        )
        self.initialize(self.tail[-1])

    def initialize(self, module):
        if isinstance(module, nn.Conv2d):
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)

    def forward(self, x, t):
        temb = self.time_embedding(t)
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        for layer in self.middleblocks:
            h = layer(h, temb)
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)
        return h