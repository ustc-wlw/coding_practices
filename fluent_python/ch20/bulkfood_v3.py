'''
属性描述符：实现 __set__、__get__、__del__
'''
class Quantity:
    def __init__(self, name) -> None:
        print('instance Quantity for ', name)
        self.storage_name = name

    def __set__(self, instance, value):
        print('call __set__ for ', self.storage_name)
        if value > 0:
            instance.__dict__[self.storage_name] = value
        else:
            raise ValueError('{} can not less than 0'.format(self.storage_name))

class LineItem:
    '''
    使用属性描述符创建类属性, 管理LineItem中托管属性: weight price
    '''
    weight = Quantity('weight')
    price = Quantity('price')

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
print('vars: ', vars(li))