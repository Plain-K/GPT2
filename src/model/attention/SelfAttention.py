import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self,d_in,d_out,qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in,d_out,bias=False)
        self.W_key   = nn.Linear(d_in,d_out,bias=False)
        self.W_value = nn.Linear(d_in,d_out,bias=False)

    def forward(self,x):
        querys = self.W_query(x)
        keys   = self.W_key(x)
        values = self.W_value(x)

        attn_scores = querys @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5,dim=1)

        context_vec = attn_weights @values
        return context_vec


d_in  = 3
d_out = 2
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
torch.manual_seed(789)
sa = SelfAttention(d_in,d_out)
print(sa(inputs))

#实例一个自注意力块
queries = sa.W_query(inputs)
keys = sa.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5,dim=-1)
print(attn_weights)

#构造一个mask掩藏矩阵
context_length = attn_weights.shape[0]
mask_simple = torch.tril(torch.ones(context_length,context_length))
print(mask_simple)

"考虑一下不同的掩藏的时机和方法，为什么会选择-inf呢"
"可能是出于优化目的，也许使用-inf来softmax的速度会更快"
masked_simple = mask_simple * attn_weights #三角矩阵，左右乘都可以
print(masked_simple)

mask = torch.triu(torch.ones(context_length,context_length),diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)

attn_weights = torch.softmax(masked / keys.shape[-1]**0.5,dim=-1)
print(attn_weights)

"dropout"
"1/（1-droprate），在dropout后进行该比例的缩放"












