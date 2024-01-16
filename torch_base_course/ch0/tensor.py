import numpy as np
import torch

def test1():
    arr = np.array([[1,2,3], [4,5,6]])
    t = torch.from_numpy(arr)

    print(f'before change: {arr}')

    arr[0][0] = 0
    print('numpy array: ', arr)
    print('tensor: ', t)

    t[0][0] = -2
    print('numpy array: ', arr)
    print('tensor: ', t)

def test2():
    t = torch.full((3,3), 8)
    print(t)

def test3():
    # t = torch.arange(1, 6, step=1)
    # t = torch.linspace(0,1, steps=20)
    # t  =torch.normal(0,1, size=(4, ))
    mean = torch.arange(1, 5, dtype=torch.float)
    t = torch.normal(mean, 1)
    print(t)

def test4():
    # t = torch.randn((2,3))
    # t  =torch.rand(2,3)
    t = torch.randint(1,5, size=(2,3))
    print(t)


if __name__=='__main__':
    # test1()
    # test2()
    test4()