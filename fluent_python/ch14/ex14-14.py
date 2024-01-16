'''
用于过滤功能的生成器方法
'''
import itertools

def vowel(c):
    return c.lower() in 'aeiou'

ss = 'Aardvark'

gen = filter(vowel, ss)
print('filter: ', list(gen)) # ['A, a, a']

gen = itertools.filterfalse(vowel, ss)
print('filterfalse: ', list(gen))

gen = itertools.dropwhile(vowel, ss)
print('dropwhile: ', list(gen)) # ['r', 'd', 'v', 'a', 'r', 'k']

gen = itertools.compress(ss, (1,0,1,0,1))
print('compress: ', list(gen))

gen = itertools.islice(ss, 4)
print('gen: ', gen)
print('islice: ', list(gen))

gen = itertools.islice(ss, 0, 4, 2) # start, stop, step
print('islice: ', list(gen))
