'''
为任意对象做浅拷贝和深拷贝
'''
import copy

class Bus:
    def __init__(self, passengers=None) -> None:
        if passengers is None:
            self.passengers = []
        else:
            self.passengers = list(passengers)

    def pick(self, name):
        self.passengers.append(name)

    def drop(self, name):
        self.passengers.remove(name)

def test1():
    passengers = ['Alice', 'Bill', 'Claire', 'David']
    bus1 = Bus(passengers)
    bus2 = copy.copy(bus1)
    bus3 = copy.deepcopy(bus1)
    print('id for bus1 , bus2 and bus3: ', id(bus1), id(bus2), id(bus3))

    bus1.drop('Bill')
    print(bus1.passengers, bus2.passengers, bus3.passengers)
    print('id for passengers for copy bus:', id(bus1.passengers), id(bus2.passengers))

if __name__=='__main__':
    test1()
