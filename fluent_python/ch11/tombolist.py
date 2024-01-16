'''
Tombola的虚拟子类, 使用Tombola.register()方法注册或者
@Tombola.register装饰器注册
'''
from random import randrange

from tombola import Tombola

@Tombola.register
class TomboList(list):

    def pick(self):
        if self:
            position = randrange(len(self))
            return self.pop(position)
        else:
            raise LookupError('pick from empty TomboList')

    def load(self, iterable):
        super().extend(iterable)

    def loaded(self):
        return bool(self)

    def inspect(self):
        return tuple(sorted(self))

# Tombola.register(TomboList)

print(issubclass(TomboList, Tombola))
print(isinstance(TomboList(), Tombola))

## 类属性: __mro__ 方法解析顺序
print(TomboList.__mro__) 
# (<class '__main__.TomboList'>, <class 'list'>, <class 'object'>)