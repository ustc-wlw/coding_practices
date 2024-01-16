'''
特性工厂函数
'''
def quantity(storage_name):
    def qty_getter(instance):
        print('call {} getter'.format(storage_name))
        return instance.__dict__[storage_name]

    def qty_setter(instance, value):
        print('call {} setter '.format(storage_name))
        if value > 0:
            instance.__dict__[storage_name] = value
        else:
            raise ValueError('{} can not less than 0!'.format(storage_name))

    return property(fget=qty_getter, fset=qty_setter)

class LineItem:
    weight = quantity('weight')
    price = quantity('price')

    def __init__(self, info, weight, price) -> None:
        print('------ init -----------')
        self.info = info
        self.weight = weight
        self.price = price

    def total(instance):
        print('----------------- total ----------')
        return instance.weight * instance.price

li = LineItem('catoo', 12, 6)
print(li.total())
print(vars(li))