'''
上下文管理类 实现__enter__、__exit__方法
'''
import sys

class LookingGlass:
    def __enter__(self):
        self._orig_writer = sys.stdout.write
        sys.stdout.write = self.reverse_write
        return 'This is enter return value'
        
    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        sys.stdout.write = self._orig_writer
        ## 如果返回None或者True之外的值 with中异常会向上冒泡
        if exc_type is ZeroDivisionError:
            print('Please do not divide by 0')
            return True

    def reverse_write(self, text:str):
        self._orig_writer(text[::-1])

def test1():
    with LookingGlass() as what:
        print('Hello World')
        print('what is inside of with block:', what)

    print('what is ourter of with block: ', what)

def test2():
    manager = LookingGlass()
    print(manager)

    monster = manager.__enter__()
    print(monster == 'This is enter return value')
    print(monster)

    manager.__exit__(None, None, None)
    print(monster)

if __name__=='__main__':
    test1()
    # test2()