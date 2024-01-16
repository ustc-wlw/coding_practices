'''
不同方式实现阶乘
'''

import functools
import operator

def test1():
    n = 1
    for i in range(1, 6):
        n *= i
    print("5! = ", n)
    return n

def test2():
    return functools.reduce(lambda a,b: a*b, range(1, 6))

def test3():
    return functools.reduce(operator.mul, range(1, 6))

if __name__=='__main__':
    print(test1() == test2())
    print(test2() == test3())
    print('result: ', test1())