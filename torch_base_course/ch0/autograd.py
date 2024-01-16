import torch

w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

def test1():
    a = torch.add(w, x) # 3
    b = torch.add(w, 1) # 2
    y = torch.mul(a, b)

    w.add_(1)
    y.backward(retain_graph=True)
    print(w.grad, x.grad)

    y.backward()
    print(w.grad, x.grad)


def test2():
    x = torch.tensor([3.0], requires_grad=True)
    y = torch.pow(x, 2)
    grad1 = torch.autograd.grad(y, x, create_graph=True)
    print('grad1: ', grad1)
    grad2 = torch.autograd.grad(grad1, x)
    print('grad2: ', grad2)

if __name__=='__main__':
    test1()