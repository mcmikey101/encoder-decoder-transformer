import torch
import torch.nn as nn
import math

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model has to be divisible by h cuz we splittin the vectors into equal parts for each head to handle"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(key, query, value, mask, dropout: nn.Dropout):
        d_k = key.shape[-1]
        #transposing the last two dims in key
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            #replace every value where mask == 0 with -1e9
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim = -1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return attention_scores @ value, attention_scores

    #we have seemingly only one matrix for key, query and value, but actually it can be interpreted as separate matrices for each head concatenated
    #when we split the output of the dot product, we essentially recover the result of each individual head projection
    #so each head works with its own section of the higher dimensional vector space

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        #transposing to get (batch, h, seq_len, d_k) - easier to interpret, having a batch of sequences, each sequence split into heads, each head containing vectors of dimensions d_k
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(key, query, value, mask, self.dropout)

        # transposing to get (batch, h, seq_len, d_k) back into (batch, seq_len, h, d_k) like just before splitting
        #contiguous means inplace i think
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)


        
