import time

DEFAULT_FMT = '[{elapsed:0.8f}s] {name}({args}) -> {result}'

'''
装饰器工厂函数
'''
def clock(fmt=DEFAULT_FMT):
    def decorate(func:callable):
        def clocked(*args):
            t0 = time.perf_counter()
            ret = func(*args)
            elapsed = time.perf_counter() - t0
            name = func.__name__
            args = ', '.join(repr(arg) for arg in args)
            result = repr(ret)
            print(fmt.format(**locals()))
            return ret
        
        return clocked
    return decorate

def test1():
    @clock()
    def snooze(seconds):
        time.sleep(seconds)

    for i in range(3):
        snooze(.123)
    

if __name__=='__main__':

    @clock('{name}({args}) dt={elapsed:0.3f}s')
    def snooze(seconds):
        time.sleep(seconds)

    for i in range(3):
        snooze(.123)
    