import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedder = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model)

    def forward(self, x):
        return self.embedder(x) * math.sqrt(self.d_model) # the paper does the multiplication. For "numerical stability" or smth i guess
    

