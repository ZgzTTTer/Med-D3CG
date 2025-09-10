import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from .base import BaseUNet
from .blocks import ResBlock

class MidAttnUNet(BaseUNet):
    def forward(self,x,t):
        temb=self.time_embedding(t)
        h=self.head(x)
        hs=[h]

        for layer in self.downblocks:
            h=layer(h,temb)
            hs.append(h)

        attention_map=None
        for i,layer in enumerate(self.middleblocks):
            h=layer(h,temb)
            if i==0 and hasattr(layer.attn,'attention_map'):
                attention_map=layer.attn.attention_map

        for layer in self.upblocks:
            if isinstance(layer,ResBlock):
                h=torch.cat([h,hs.pop()],dim=1)
            h=layer(h,temb)

        h=self.tail(h)
        return h, attention_map

class UpAttnUNet(BaseUNet):
    def forward(self,x,t):
        temb=self.time_embedding(t)
        h=self.head(x)
        hs=[h]
        for layer in self.downblocks:
            h=layer(h,temb)
            hs.append(h)
        for layer in self.middleblocks:
            h=layer(h,temb)

        last_up_attention_map=None
        for layer in self.upblocks:
            if isinstance(layer,ResBlock):
                h=torch.cat([h,hs.pop()],dim=1)
            h=layer(h,temb)
            if isinstance(layer,ResBlock) and hasattr(layer.attn,'attention_map'):
                last_up_attention_map = layer.attn.attention_map

        h=self.tail(h)
        return h, last_up_attention_map