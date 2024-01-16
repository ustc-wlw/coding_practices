'''
规约生成器函数
'''

print(all([1,2,3]))

g = all([0,2,3])
print(g)

print(all([]))

g = (n for n in [0, 0, 7, 6])
print(any(g))
print('next value: ', next(g))
