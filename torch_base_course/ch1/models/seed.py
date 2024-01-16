import torch

torch.manual_seed(42)

x = torch.rand(2,3)

# torch.random.manual_seed(42)
y = torch.rand(2,3)

print(x, y)