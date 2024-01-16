import torch

# cat
def test1():
    t = torch.ones(2,3)
    t1 = torch.cat([t, t], dim=0)
    t2 = torch.cat([t, t], dim=1)
    print(f'cat on dim 0: {t1}')
    print(f'cat on dim 1: {t2}')

# stack
def test2():
    t = torch.ones(2,3)
    t1 = torch.stack([t, t], dim=2)
    print('stack on dim 2: ', t1, t1.shape)

def test3():
    t = torch.ones(2,7)
    list_of_tensors = torch.chunk(t, 3, dim=1)
    for i, t in enumerate(list_of_tensors):
        print(f'{i+1}, {t}, {t.shape}')

def test4():
    t = torch.ones(2,5)
    # ret = torch.split(t, 3, dim=1)
    ret = torch.split(t, [2,1,2], dim=1)
    for i, t in enumerate(ret):
        print(f'{i+1}, {t}, {t.shape}')

def test5():
    t = torch.randint(0,9, size=(3,3))
    print('roigin tensor: ', t)
    index = torch.tensor([0,2])
    print(index.dtype)
    t_select = torch.index_select(t, dim=0, index=index)
    print(f'index_select tensor: {t_select}')

def test6():
    t = torch.randint(0,9, size=(3,3))
    mask = t.le(5)
    print(f'mask is {mask}')
    ret = torch.masked_select(t, mask)
    print(ret)

def test7():
    t = torch.randperm(8)
    print('origin t: ', t)
    t_reshape = torch.reshape(t, (-1, 2, 2))
    print('after reshape: ', t_reshape)
    print(id(t.data), id(t_reshape.data))

def test8():
    t = torch.rand(2,3,4)
    t_transpose = torch.transpose(t, 1, 2)
    print(f't shape {t.shape}, t_transpose shape {t_transpose.shape}')

def test9():
    t = torch.rand((1, 2, 3, 1))
    print('t shape: ', t.shape)
    t_sq = torch.squeeze(t, 0)
    print('t_sq shape: ', t_sq.shape)
    t_usq = torch.unsqueeze(t, 0)
    print('unsquzee shape: ', t_usq.shape)

if __name__=="__main__":
    # test1()
    # test2()
    test9()