'''
用于映射的生成器方法
从输入的可迭代对象的各个元素产出一个元素
如果输入来自多个可迭代对象，第一个可迭代对象到头后就结束
'''
import itertools
import operator

sample = [5, 4, 2, 8, 7, 6, 3, 0, 9, 1]

gen = itertools.accumulate(sample)
print('gen: ', gen)
print(next(gen), list(gen))

gen = itertools.accumulate(sample, min)
print('accumulate.min: ', list(gen))

gen = itertools.accumulate(sample, operator.mul)
print('accumulate.mul', list(gen))

gen = enumerate('abcdefg', 1)
for x,y in gen:
    print(x,y)

gen = map(operator.mul, range(1,11), range(1,11))
print('map.mul: ', list(gen))

gen = map(lambda a,b: (a,b), 'abcd', range(3))
print('map.lambda: ', list(gen))

gen = itertools.starmap(operator.mul, enumerate('abc', 1))
print('starmap: ', list(gen))