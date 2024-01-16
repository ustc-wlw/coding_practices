'''
使用特性验证属性
'''
class LineItem:
    def __init__(self, name, weight, price) -> None:
        self.name = name
        self.weight = weight
        self.price = price

    def total(self):
        return self.weight * self.price

li = LineItem('Tomato', 10, 6.9)
print('dir(obj): ', dir(li))
print('__dict__: ', li.__dict__)
print('__class___: ', li.__class__)