'''
定义属性描述符的子类, 实现自定义功能的属性描述符类
'''
import abc

class AutoStorage:
    __counter = 0

    def __init__(self) -> None:
        cls = self.__class__
        self.storage_name = '_{}#{}'.format(cls.__name__, cls.__counter)
        cls.__counter += 1

    def __get__(self, instance, owner):
        if instance:
            return getattr(instance)
        else:
            return self
    
    def __set__(self, instance, value):
        setattr(instance, self.storage_name, value)

'''
模板方法设计模式
'''
class Validated(AutoStorage):
    @abc.abstractmethod
    def validate(self, instance, value):
        '''return validated value or raise ValueError'''

    def __set__(self, instance, value):
        value = self.validate(instance, value)
        super().__set__(instance, value)
        

class Quantity(Validated):
    def validate(self, instance, value):
        if value > 0:
            return value
        else:
            raise ValueError('value must greater than 0')

class NonBlank(Validated):
    def validate(self, instance, value):
        if value:
            return value
        else:
            raise ValueError('value must be not empty')