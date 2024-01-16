def make_averager():
    count = 0
    total = 0

    def averager(new_value):
        count += 1
        total += new_value
        return total / count

    return averager

def make_averager_nolocal():
    count = 0
    total = 0

    def averager(new_value):
        nonlocal count, total
        count += 1
        total += new_value
        return total / count

    return averager


def test1():
    avg = make_averager()
    print('avg type: ', type(avg))
    print(avg.__code__.co_argcount)
    print(avg.__code__.co_varnames)
    print(avg.__code__.co_freevars)
    print('closure: ', avg.__closure__)
    # print('closure cell contents: ', avg.__closure__[0].cell_contents)
    print(avg(10))

def test2():
    avg = make_averager_nolocal()
    print('avg type: ', type(avg))
    print('arg count: ', avg.__code__.co_argcount)
    print('var names: ', avg.__code__.co_varnames)
    print('free vars: ', avg.__code__.co_freevars)
    print('closure: ', avg.__closure__)
    print('closure cell contents: ', avg.__closure__[0].cell_contents)

    print(avg(10))
    print(avg(11))
    print(avg(12))

if __name__=='__main__':
    test1()

    # test2()