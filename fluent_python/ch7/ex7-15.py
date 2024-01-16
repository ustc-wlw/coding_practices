import time
import functools

def clock(func:callable):
    ## copy func attributes to clocked function
    @functools.wraps(func)
    def clocked(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        name = func.__name__
        arg_str = ', '.join([repr(arg) for arg in args])
        print('[%0.8fs] %s(%s) --> %r' % (elapsed, name, arg_str, result))
        return result

    return clocked

@clock
def snooze(seconds):
    time.sleep(seconds)

@clock
def factorial(n):
    return 1 if n < 2 else n * factorial(n-1)

def test1():
    print('*' * 40, 'Calling snooze(.123)')
    snooze(.123)
    print('*' * 40, 'Calling factorial(6)')
    print('6! = ', factorial(6))

def test2():
    print('snooze name: ', snooze.__name__)
    print('factorial name ', factorial.__name__)

if __name__=='__main__':
    # test1()
    test2()