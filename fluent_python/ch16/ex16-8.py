import inspect

class DemoException(Exception):
    '''demo Exception'''

def demo_exc_handling():
    print('-> coroutine started....')
    try:
        while True:
            try:
                x = yield
            except DemoException:
                print('DemoException hanled, and generate continue....')
            else:
                print('receive data is {!r}'.format(x))
    finally:
        print('-> coroutine ended!!')

def test1():
    gen = demo_exc_handling()
    print(gen)
    print(inspect.getgeneratorstate(gen))

    next(gen)
    print(inspect.getgeneratorstate(gen))
    gen.send(10)

    gen.send(20)

    gen.throw(DemoException)
    print(inspect.getgeneratorstate(gen))

    gen.throw(ZeroDivisionError)
    print(inspect.getgeneratorstate(gen))

if __name__=='__main__':
    test1()