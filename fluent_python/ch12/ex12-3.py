'''
不要子类化内置类型,例如: list, dict
因为内置类型的方法不会调用子类覆写的方法
如果需要, 子类化collections中的抽象基类如 UserList UserDict UserString
'''
import collections

class DoppeDict(collections.UserDict):
    def __setitem__(self, key, item) -> None:
        return super().__setitem__(key, [item] * 3)

class DoppeDict2(dict):
    def __setitem__(self, key, item) -> None:
        return super().__setitem__(key, [item] * 3)


class AnswerDict(collections.UserDict):
    def __getitem__(self, key):
        return 'const value'

class AnswerDict2(dict):
    def __getitem__(self, key):
        return 'const value'

def test1():
    dd = DoppeDict(one=1)
    print(dd)

    dd['two']=2
    dd.update(three=3)
    print(dd)

    ad = AnswerDict(a='foo')
    print(ad['a'])

def test2():
    dd = DoppeDict2(one=1)
    print(dd)

    dd['two']=2
    dd.update(three=3)
    print(dd)

    ad = AnswerDict2(a='foo')
    print(ad['a'])

    d = {}
    d.update(ad)
    print(d)

if __name__=='__main__':
    # test1()
    test2()