class LineItem:
    def __init__(self, name, weight, price) -> None:
        self.name = name
        self.weight = weight
        self.price = price

    @property
    def weight(self):
        return self.__weight

    @weight.setter
    def weight(self, value):
        if value > 0:
            self.__weight = value
        else:
            raise ValueError('weight can not less than zero')

li = LineItem('rice', 11, 6.7)
print('instance __dict__: ', li.__dict__)
print('class __dict__: ', LineItem.__dict__)

li.weight = -10