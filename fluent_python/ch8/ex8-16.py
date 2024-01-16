'''
del删除对象引用及垃圾回收机制
'''
import weakref

s1 = {1,2,3}

s2 = s1

def bye():
    print('Gone with wind')

ender = weakref.finalize(s1, bye)
print(ender.alive)

del s1
print(ender.alive)

s2 = 'spam' # no reference available for {1,2,3}
print(ender.alive) # False


