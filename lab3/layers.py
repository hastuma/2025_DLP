import torch.nn as nn
import torch
import math

#TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.q_linear = nn.Linear(self.dim, self.dim)
        self.k_linear = nn.Linear(self.dim, self.dim)
        self.v_linear = nn.Linear(self.dim, self.dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(self, x):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''
        batch, num_image_tokens, dim = x.size()
        # fc layer 
        q = self.q_linear(x).view(batch, -1, self.num_heads, self.dim // self.num_heads).transpose(1,2)
        k = self.k_linear(x).view(batch, -1, self.num_heads, self.dim // self.num_heads).transpose(1,2)
        v = self.v_linear(x).view(batch, -1, self.num_heads, self.dim // self.num_heads).transpose(1,2)
        # scaled dot-product attention

        scores = torch.matmul(q, k.transpose(2, 3)) /  math.sqrt(self.dim // self.num_heads)
        scores = nn.functional.softmax(scores, dim=3)
        scores = self.attn_drop(scores)
        output = torch.matmul(scores, v)
        # concat and fully connected layer 
        output = output.transpose(1,2).contiguous().view(batch, -1, self.num_heads*self.dim // self.num_heads)
        output = self.proj(output)
        return output
        # raise Exception('TODO1!')

class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    