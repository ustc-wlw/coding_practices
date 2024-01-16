from abc import ABC, abstractmethod
from collections import namedtuple

Customer = namedtuple('Customer', 'name fidelity')

class LineItem:
    def __init__(self, product:str, quantity:int, price:float) -> None:
        self.product = product
        self.quantity = quantity
        self.price = price

    def total(self):
        return self.price * self.quantity

class Order:
    def __init__(self, customer:Customer, cart, promotion:callable=None) -> None:
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
            discount = self.promotion(self)
        return self.total() - discount

    def __repr__(self) -> str:
        fmt = '<Order total: {:.2f} due: {:.2f}>'
        return fmt.format(self.total(), self.due())

def fidelity_promo(order:Order):
    '''积分1000及以上提供5%折扣'''
    return order.total() * .05 if order.customer.fidelity >= 1000 else 0

def bulkItem_promo(order:Order):
    '''单个商品为20或者以上提供10%折扣'''
    discount = 0
    for item in order.cart:
        if item.quantity >= 20:
            discount += item.total() * .1
    return discount

def largeOrder_promo(order:Order):
    '''订单中不同商品数量达到10个或者以上时提供7%折扣'''
    distinct_item = {item.product for item in order.cart}
    return order.total() * .07 if len(distinct_item) >= 10 else 0

def best_promo(order:Order):
    promos = [globals()[name] for name in globals() 
                                if name.endswith('_promo')
                                and name != 'best_promo']
    print('all promotions are: ', promos)
    return max((promo(order) for promo in promos))

joe = Customer('joe', 0)
ann = Customer('Ann', 1100)
cart = [LineItem('banana', 4, .5),
        LineItem('apple', 10, 1.5),
        LineItem('watermellon', 5, 5.0)]

def test_promote():
    print(Order(joe, cart, fidelity_promo))
    print(Order(ann, cart, fidelity_promo))

def test_promotes2():
    print(Order(joe, cart, best_promo))

if __name__=='__main__':
    test_promotes2()
    # test_promote()