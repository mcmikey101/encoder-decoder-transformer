import torch
import torch.nn as nn
from blocks.encoder import Encoder
from blocks.decoder import Decoder
from preparation.embedding import InputEmbedding
from preparation.pos_encoding import PositionalEncoding
from blocks.projection import ProjectionLayer

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_emb: InputEmbedding, tgt_emb: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection = projection
        
    def encode(self, x, src_mask):
        x = self.src_emb(x)
        x = self.src_pos(x)
        x = self.encoder(x, src_mask)
        return x
    
    def decode(self, x, encoder_output, src_mask, tgt_mask):
        x = self.tgt_emb(x)
        x = self.tgt_pos(x)
        x = self.decoder(x, encoder_output, src_mask, tgt_mask)
        return x

    def project(self, x):
        return self.projection(x)
    