'''
特性会覆盖同名实例属性
'''
class Demo:
    data = 'the class data attribute'

    @property
    def prop(self):
        return 'the property data'

obj = Demo()

def test1():
    '''
    实例属性会覆盖同名类属性
    '''
    print('obj attributes: ', vars(obj))
    print('class attributes: ', Demo.__dict__)
    print(obj.data)

    obj.data = 'instance attribute data'
    print('instance attr: ', vars(obj))
    print(obj.data)

    print('Class attri: ', Demo.data)

def test2():
    '''
    实例属性不会覆盖特性
    '''
    print(Demo.prop)
    print(obj.prop)
    print('instance dict: ',obj.__dict__)
    # obj.prop = 'instance property' # error
    # print(obj.prop)
    obj.__dict__['prop'] = 'instance property'
    print('instance dict: ',obj.__dict__)
    print(obj.prop)

    Demo.prop = 'repalce prop with a str'
    print(obj.prop)
    print(Demo.prop)

def test3():
    '''
    新添加的特性覆盖实例属性
    '''
    obj.data = 'instance data'
    print(vars(obj))
    print(Demo.data)
    Demo.data = property(lambda self : 'a new added property')
    print('Class dict: ', Demo.__dict__)
    print(obj.data)

    del Demo.data
    print(obj.data)


if __name__=='__main__':
    # test1()
    # test2()
    test3()