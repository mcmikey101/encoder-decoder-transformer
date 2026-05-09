import torch
import torch.nn as nn
from substructures.attention import MultiHeadAttentionBlock
from substructures.residual import ResidualConnection
from substructures.ffn import FeedForwardBlock
from substructures.layer_norm import LayerNorm


class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)

    #the mask is for removing padding tokens like <pad>
    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
            
        return self.norm(x)


