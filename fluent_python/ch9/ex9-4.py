'''
类方法和静态方法
'''
class Demo:
    @classmethod
    def klassmeth(*args):
        return args

    @staticmethod
    def statmeth(*args):
        return args

print(Demo.klassmeth()) # (<class '__main__.Demo'>,)

print(Demo.klassmeth('hello')) # (<class '__main__.Demo'>, 'hello')

print(Demo.statmeth()) # ()