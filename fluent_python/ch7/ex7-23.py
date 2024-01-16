# set
registry = set()

'''
装饰器工厂函数
'''
def register(activate=True):
    def decorate(func:callable):
        print('running register(activate=%s) -> decorate(%s)' % (activate, func))
        if activate:
            registry.add(func)
        else:
            registry.discard(func)

        return func

    return decorate

@register(activate=False)
def f1():
    print('running f1()')

@register()
def f2():
    print('running f2()')

def f3():
    print('runing f3()')

def test1():
    print('main and test1 runing.....')
    print(registry)
    f1()
    f2()
    # f3()

    register()(f3)

if __name__=='__main__':
    test1()
