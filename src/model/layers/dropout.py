import torch.nn
import torch

torch.manual_seed(123)
example = torch.ones(6,6)
dropout = torch.nn.Dropout(0.5)
print(dropout(example))
