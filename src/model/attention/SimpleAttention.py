import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

#求第二个token的上下文向量
query =inputs[1]

#注意力分数
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(query,x_i)
print(attn_scores_2)

#->注意力权重
attn_weights_2 =torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

#加权求和得到上下文向量
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print(context_vec_2)

#注意力矩阵
#attn_scores = torch.empty(inputs.shape)错误，矩阵大小应为n^2
attn_scores = torch.empty(6,6)
attn_scores = inputs @ inputs.T
attn_weights = torch.softmax(attn_scores,dim=1)#注意到scores是一个对角矩阵，但只在列维度归一化后，weight就不再是对角矩阵了
context_vecs = attn_weights @ inputs
print(context_vecs)

