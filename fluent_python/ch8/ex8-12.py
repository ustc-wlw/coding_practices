'''
不能使用可变类型作为参数默认值
'''
class HauntBus:
    '''如果使用可变类型做为参数默认值会导致意外的结果'''
    def __init__(self, passengers=[]) -> None:
        self.passengers:list = passengers

    def pick(self, name):
        self.passengers.append(name)

    def drop(self, name):
        self.passengers.remove(name)

def test1():
    bus1 = HauntBus(['Alice', 'Bill'])
    print(bus1.passengers)
    bus1.pick('Charlie')
    bus1.drop('Alice')
    print('after pick Charlie and drop Alice, ', bus1.passengers)

    bus2 = HauntBus()
    bus2.pick('Carrie')
    print('bus2 passengers: ', bus2.passengers)

    bus3 = HauntBus()
    print('bus3 passengers after init: ', bus3.passengers)
    print('bus2 and bus3 passengers id: ', id(bus3.passengers), id(bus2.passengers), 
            bus3.passengers is bus2.passengers)

    bus3.pick('Dave')
    print('bus2 passengers: ', bus2.passengers)
    print('bus1 passengers: ', bus1.passengers)

def test2():
    print(dir(HauntBus.__init__))
    print(HauntBus.__init__.__defaults__)

if __name__=='__main__':
    test2()
    # test1()