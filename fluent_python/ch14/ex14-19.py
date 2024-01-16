'''
把输入的各个元素扩展为多个输出元素的生成器方法
'''
import itertools

gen = itertools.count()
print(next(gen))
print(next(gen), next(gen), next(gen))

gen = itertools.islice(itertools.count(), 4)
print('count slice: ', list(gen))

gen = itertools.cycle('ABC')
print(next(gen), next(gen), next(gen),next(gen), next(gen), next(gen))

gen = itertools.islice(itertools.cycle('ABC'), 6)
print('islice cycle: ', list(gen))

gen = itertools.repeat(5, 4)
# print('repeat: ', next(gen), next(gen), next(gen),next(gen), next(gen), next(gen))

## 排列组合
gen = itertools.combinations('ABC', 2)
print('combinations: ', list(gen))

gen = itertools.combinations_with_replacement('ABC', 2)
print('combinations_with_replacement: ', list(gen))

gen = itertools.permutations('ABC', 2)
print('permutations: ', list(gen))
