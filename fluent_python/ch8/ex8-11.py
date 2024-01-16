def f(a, b):
    a += b
    print('after f run, a is ', a)

def test1():
    x = 1
    y = 2
    f(x, y)
    print('x , y: ', x, y)

def test2():
    x = [1,2]
    y = [3,4]
    f(x, y)
    print('x , y: ', x, y)

def test3():
    x = (1,2)
    y = (3,4)
    f(x, y)
    print('x , y: ', x, y)


if __name__=='__main__':
    test3()
    # test2()
    # test1()