from functools import wraps
import inspect
'''
装饰器：向前执行到第一个`yield`表达式，预激`func`
func: 生成器函数
'''
def coroutine(func):
    @wraps(func)
    def decorator(*args, **kvargs):
        gen = func(*args, **kvargs)
        print('gen state after call func: ', inspect.getgeneratorstate(gen))
        next(gen)
        print('gen state after next: ', inspect.getgeneratorstate(gen))
        return gen
    return decorator