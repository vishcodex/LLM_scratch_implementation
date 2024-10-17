import torch
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])

for i, x_i in enumerate(inputs):
 print("results ",i, x_i, query)
 attn_scores_2[i] = torch.dot(x_i, query)

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

contexxt_vec_2 = torch.zeros(query.shape)

for i, x_i in enumerate(inputs):
 print(i, contexxt_vec_2, attn_weights_2[i], x_i, attn_weights_2[i]* x_i)
 contexxt_vec_2 = contexxt_vec_2 + attn_weights_2[i] * x_i

print(inputs.shape)
print(inputs.shape[0])
print("attn_scores_2 :",attn_scores_2)
print("attn_weights_2 :", attn_weights_2)
print("contexxt_vec_2 :", contexxt_vec_2)