import torch
import torch.nn as nn
from blocks.encoder import Encoder, EncoderBlock
from blocks.decoder import Decoder, DecoderBlock
from blocks.projection import ProjectionLayer
from blocks.transformer import Transformer
from substructures.attention import MultiHeadAttentionBlock
from substructures.ffn import FeedForwardBlock
from preparation.embedding import InputEmbedding
from preparation.pos_encoding import PositionalEncoding

def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model = 128, N = 1, h = 1, dropout = 0.1, d_ff = 256):
    src_emb = InputEmbedding(d_model=d_model, vocab_size=src_vocab_size)
    tgt_emb = InputEmbedding(d_model=d_model, vocab_size=tgt_vocab_size)

    src_pos = PositionalEncoding(d_model=d_model, seq_length=src_seq_len, dropout=dropout)
    tgt_pos = PositionalEncoding(d_model=d_model, seq_length=tgt_seq_len, dropout=dropout)

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttentionBlock(d_model=d_model, h=h, dropout=dropout)
        feed_forward_block = FeedForwardBlock(d_model=d_model, d_ff=d_ff, dropout=dropout)
        encoder_block = EncoderBlock(self_attention_block=encoder_self_attention, feed_forward_block=feed_forward_block, dropout=dropout)
        encoder_blocks.append(encoder_block)
    
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention = MultiHeadAttentionBlock(d_model=d_model, h=h, dropout=dropout)
        decoder_cross_attention = MultiHeadAttentionBlock(d_model=d_model, h=h, dropout=dropout)
        feed_forward_block = FeedForwardBlock(d_model=d_model, d_ff=d_ff, dropout=dropout)
        decoder_block = DecoderBlock(self_attention_block=decoder_self_attention, cross_attention_block=decoder_cross_attention, feed_forward_block=feed_forward_block, dropout=dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection = ProjectionLayer(d_model=d_model, vocab_size=tgt_vocab_size)

    transformer = Transformer(encoder=encoder, decoder=decoder, src_emb=src_emb, tgt_emb=tgt_emb, src_pos=src_pos, tgt_pos=tgt_pos, projection=projection)

    #parameter initialization (voodoo shit)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

def build_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"], config["d_model"])
    return model