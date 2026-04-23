import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

#定义输入-输出维度
x_2 = inputs[1]#取第二个token的嵌入向量
d_in = inputs.shape[1]#嵌入维度
d_out = 2

#定义qkv
torch.manual_seed(123)
W_q = torch.nn.Parameter(torch.rand(d_in,d_out),requires_grad=False)
W_k = torch.nn.Parameter(torch.rand(d_in,d_out),requires_grad=False)
W_v = torch.nn.Parameter(torch.rand(d_in,d_out),requires_grad=False)

#计算第二个token的qkv，列组合
q_2 = x_2 @ W_q
k_2 = x_2 @ W_k
v_2 = x_2 @ W_v
print(q_2)

#计算整个输入序列的qkv
qs = inputs @ W_q
ks = inputs @ W_k
vs = inputs @ W_v
print(qs.shape,ks.shape,vs.shape)

#计算q_2的注意力分数
attn_scores_2 = q_2 @ ks.T
print(attn_scores_2)

d_k = ks.shape[1]
attn_weights_2 = torch.softmax(attn_scores_2/d_k ** 0.5,dim = -1)

context_vec_2 = attn_weights_2 @ vs
print(context_vec_2)





