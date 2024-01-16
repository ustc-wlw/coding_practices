'''
属性描述符类实现
'''
class Quantity:
    '''
    类属性，用于计数
    '''
    __counter = 0
    def __init__(self) -> None:
        cls = self.__class__
        self.storage_name = '_{}#{}'.format(cls.__name__, cls.__counter)
        print('storage name is ', self.storage_name)
        cls.__counter += 1

    def __set__(self, instance, value):
        if value > 0:
            setattr(instance, self.storage_name, value)
        else:
            raise ValueError('value must greater than 0')

    def __get__(self, instance, ower):
        if instance:
            return getattr(instance, self.storage_name)
        else:
            return self

class LineItem:
    '''
    使用属性描述符创建类属性, 管理LineItem中托管属性: weight price
    '''
    weight = Quantity()
    price = Quantity()

    def __init__(self, name, weight, price) -> None:
        print('instance LineItem ..........')
        self.name = name
        self.weight = weight
        self.price = price

    def total(self):
        return self.weight * self.price

print('test begain ..............')
li = LineItem('tamatoo', 11, 12)
print('total: ', li.total(), li.weight, li.price)
print('obj vars: ', vars(li))
print('class vars: ', vars(LineItem))


## 从类获取类属性
print(LineItem.weight)