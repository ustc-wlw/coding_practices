'''
覆盖型描述符类与非覆盖型描述符
'''
class Overriding:
    def __set__(self, instance, value):
        print('----> set', self, instance, value)

    def __get__(self, instance, owner):
        print('<--- get', self, instance, owner)

class OverridingNoGet:
    def __set__(self, instance, value):
        print('----> set', self, instance, value)

class NoOverriding:
    def __get__(self, instance, owner):
        print('<--- get', self, instance, owner)

''''
托管类
'''
class Managed:
    overriding = Overriding()
    over_not_get = OverridingNoGet()
    no_over = NoOverriding()

    def spwan(self):
        print('here is a method')

obj = Managed()

def test1():
    obj.overriding
    print(obj.over_not_get) ## get self
    obj.no_over

    print(Managed.__dict__)
    Managed.overriding, Managed.no_over

    obj.overriding = 7

    obj.no_over = 8
    print(obj.no_over)
    print(vars(obj))
    Managed.no_over

def test2():
    obj.no_over
    obj.no_over = 6
    print(obj.no_over)

if __name__=='__main__':
    # test1()
    test2()
    