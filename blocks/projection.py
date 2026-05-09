import torch
import torch.nn as nn

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        #turning (batch, seq_len, d_model) into (batch, seq_len, vocab_size)
        return torch.log_softmax(self.projection(x), dim=-1)
