def deco(func: callable):
    def inner(*args, **kwargs):
        print('runing inner and decorate for func ', func.__name__)
    return inner

@deco
def target():
    print('runing target func')

target()
print('target: ', target)