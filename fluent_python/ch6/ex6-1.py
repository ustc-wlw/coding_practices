from abc import ABC, abstractmethod
from collections import namedtuple

Customer = namedtuple('Customer', 'name fidelity')

class LineItem:
    def __init__(self, product, quantity, price) -> None:
        self.product_name = product
        self.quantity = quantity
        self.price = price

    def total(self):
        return self.price * self.quantity

class Order:
    def __init__(self, customer:Customer, cart, promotion=None) -> None:
        self.customer = customer
        self.cart = list(cart)
        self.promotion = promotion

    def total(self):
        if not hasattr(self, '__total'):
            self.__total = sum([item.total() for item in self.cart])
        return self.__total

    def due(self):
        if self.promotion is None:
            discount = 0
        else:
            discount = self.promotion.discount(self)
        return self.total() - discount

    def __repr__(self) -> str:
        fmt = '<Order total: {:.2f} due: {:.2f}>'
        return fmt.format(self.total(), self.due())

class Promotion(ABC):
    @abstractmethod
    def discount(self, order:Order):
        pass

class FidelityPromo(Promotion):
    '''积分1000及以上提供5%折扣'''
    def discount(self, order: Order):
        return order.total() * .05 if order.customer.fidelity >= 1000 else 0

class BulkItemPromo(Promotion):
    '''单个商品为20或者以上提供10%折扣'''
    def discount(self, order: Order):
        discount = 0
        for item in order.cart:
            if item.quantity >= 20:
                discount += item.total() * .1
        return discount

class LargeOrderPromo(Promotion):
    '''订单中不同商品数量达到10个或者以上时提供7%折扣'''
    def discount(self, order: Order):
        distinct_item = {item.product for item in order.cart}
        return order.total() * .07 if len(distinct_item) >= 10 else 0

joe = Customer('joe', 0)
ann = Customer('Ann', 1100)
cart = [LineItem('banana', 4, .5),
        LineItem('apple', 10, 1.5),
        LineItem('watermellon', 5, 5.0)]

def test_promote():
    print(Order(joe, cart, FidelityPromo()))
    print(Order(ann, cart, FidelityPromo()))

if __name__=='__main__':
    test_promote()