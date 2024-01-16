'''
合并多个可迭代对象的生成器方法
'''
import itertools

gen = itertools.chain('ABC', range(3))
for c in gen:
    print(c)

gen = itertools.chain(enumerate('ABC', 1))
print('chain: ', list(gen))

gen = itertools.chain.from_iterable(enumerate('ABC', 1))
print('chain.from_iterable: ', list(gen))

gen = zip('abc', range(5))
print('zip: ', list(gen)) # [('a', 0), ('b', 1), ('c', 2)]

gen = itertools.zip_longest('abc', range(5))
print('zip_longest: ', list(gen))

gen = itertools.zip_longest('abc', range(5),fillvalue='?')
print('zip_longest with fillvalue : ', list(gen))

gen = itertools.product('ABC', range(2))
print('product, repeat=1: ', list(gen))

gen = itertools.product('ABC', range(2), repeat=2)
print('product, repeat=2: ', list(gen))

gen = itertools.product('ABC')
print('product: ', list(gen))

gen = itertools.product('ABC', repeat=2)
print('product, repeat=2: ', list(gen))




