import torch
import numpy as np

def test1():
    x: torch.Tensor = torch.arange(1, 8)
    print(x, x.shape)

    x_reshaped = x.reshape(1, 7)
    print(x_reshaped)

    x_view = x.view(1,7)
    x_view[:, 0] = 8
    print(f'x: {x}, x_view: {x_view}')

def test2():
    x = torch.arange(1, 8)
    x_stacked = torch.stack([x, x, x], dim=0)
    print(f'x_stacked: {x_stacked}')

    x_stacked_1 = torch.stack([x, x, x], dim=1)
    print(f'x_stacked_dim_1: {x_stacked_1}')

def test3():
    x = torch.randn(224, 224, 3)
    print(f'original shape: {x.shape}')
    x_p = x.permute(2, 0, 1)
    print(f'x_permuted shape: {x_p.shape}')

    array = np.arange(1.0, 8.0)
    t = torch.from_numpy(array)
    array += 1
    t_fp32 = t.type(torch.float32)
    print(f'numpy array: {array}\n tensor is {t} \n t_fp32: {t_fp32}')

    t = torch.ones(7)
    t_np = t.numpy()
    print(f'tensor is {t} \n numpy data is {t_np}')

import random
def test4():
    random_seed = 42
    torch.manual_seed(random_seed)

    random_a = torch.rand(3, 4)

    # torch.random.manual_seed(random_seed)

    random_b = torch.rand(3, 4)
    print(f'random_a: {random_a}')
    print(f'random_b: {random_b}')
    print(random_b == random_a)

if __name__=='__main__':
    # test1()
    test4()