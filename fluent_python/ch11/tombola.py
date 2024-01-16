import abc

class Tombola(abc.ABC):

    @abc.abstractmethod
    def load(self, iterable):
        '''从可迭代对象中添加元素'''

    @abc.abstractmethod
    def pick(self):
        '''随机删除元素然后将其返回
            如果实例为空应该返回LookupError
        '''

    def loaded(self):
        '''如果至少有一个元素返回True, 否则返回False'''
        return bool(self.inspect())

    def inspect(self):
        '''返回一个有序元素, 由当前元素构成'''
        items = []
        while True:
            try:
                item = self.pick()
            except LookupError:
                break
            items.append(item)
        self.load(items)
        return tuple(sorted(items))
