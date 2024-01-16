'''
装饰器的嵌套
'''
@d1
@d2
def func():
    pass

f = d1(d2(func))