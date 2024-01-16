'''
实现等差数列: 生成器函数、itertools库中count函数
'''

from itertools import count, takewhile
from fractions import Fraction

'''
鸭子类型, 可迭代对象
'''
class ArithmetricProgression:
    def __init__(self, begain, step, end=None) -> None:
        self._begain = begain
        self._step = step
        self._end = end
    
    def __iter__(self):
        result = type(self._begain + self._step)(self._begain)
        forever = self._end is None
        index = 1
        while forever or result <= self._end:
            yield result
            result = self._begain + index * self._step
            index += 1

'''
使用生成器函数（定义体中有yield关键字）实现
'''
def aritprog_gen(begain, step, end=None):
    result = type(begain + step)(begain)
    forever = end is None
    index = 0
    while forever or result <= end:
        yield result
        index += 1
        result = begain + index * step

'''
生成器函数实现：使用itertools库中方法
'''
def aritprog_gen2(begain, step , end=None):
    first = type(begain + step)(begain)
    gen = count(first, step)
    if end is not None:
        gen = takewhile(lambda x : x < end, gen)
    return gen

def test1():
    ap = ArithmetricProgression(0,1,3)

    ap = ArithmetricProgression(0, 1/3 ,1)

    ap = ArithmetricProgression(0, Fraction(1,3) ,1)

    print(list(ap))

def test2():
    ap = aritprog_gen(0,1,3)

    ap = aritprog_gen(0, 1/3 ,1)

    ap = aritprog_gen(0, Fraction(1,3) ,1)

    print(list(ap))

def test3():
    ap = aritprog_gen2(0,1,3)

    ap = aritprog_gen2(0, 1/3 ,1)

    ap = aritprog_gen2(0, Fraction(1,3) ,1)

    print(list(ap))

if __name__=='__main__':
    # test1()
    # test2()
    test3()