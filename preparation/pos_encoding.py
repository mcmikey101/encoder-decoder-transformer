import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_length: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)
        self.pos_enc = torch.zeros(seq_length, d_model) #matrix of position embeddings for each token embedding
        self.pos_vec = torch.arange(0, self.seq_length, dtype=torch.float).unsqueeze(1) #index of every token embedding 
        self.div_term = torch.exp(torch.arange(0, d_model, 2).float() * -math.log(10000.0) / d_model) # in log space for better numerical stability

        #alternate sin and cos for even and odd indexes, idk why
        #the slice means for every vector in the sequence every dimension in the position embedding becomes the product of its respective tokens position in the sequence 
        #and a term raised to the power of the index of the current dimension we are encoding
        self.pos_enc[:, 0::2] = torch.sin(self.pos_vec * self.div_term)
        self.pos_enc[:, 1::2] = torch.cos(self.pos_vec * self.div_term)

        self.pos_enc = self.pos_enc.unsqueeze(0)

        self.register_buffer("pe", self.pos_enc)

        #so the position encoding depends on the position of the embedding, index of dimension and model dimensions

    def forward(self, x):
        x = x + (self.pos_enc[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
        