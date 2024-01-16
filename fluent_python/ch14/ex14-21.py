'''
用于重排元素的生成器方法
groupby
reversed
tee
'''

import itertools

# groups = itertools.groupby('LAGAGGGLL')
groups = itertools.groupby('LLLAAGGGLL')
for key, gen in groups:
    print('key, group: ', key, list(gen))

animals = ['duck', 'eagle', 'rat', 'giraffe', 'bear']
print('after revered: ', list(reversed(animals)), animals)
animals.sort(key=len)
print('after sort: ', animals)

for key, group in itertools.groupby(animals, key=len):
    print('key, group: ', key, list(group))

