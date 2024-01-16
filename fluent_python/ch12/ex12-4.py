'''
多重继承及方法解析顺序
'''
class A:
    def ping(self):
        print('A ping: ', self)

class B(A):
    def pong(self):
        print('B pong: ', self)

class C(A):
    def pong(self):
        print('C pong: ', self)

class D(B,C):
    def ping(self):
        print('D ping start run........')
        super().ping()
        print('D ping run ends! ', self)

    def pingpong(self):
        print('D pingpong start run -----------')
        self.ping()
        super().ping()
        self.pong()
        super().pong()
        print('C.pong call ', '*'*20)
        C.pong(self)

def test1():
    d = D()
    print('d.pong run ', '-'*20)
    d.pong()
    print('C.pong run ', '=='*20)
    C.pong(d)

def test2():
    print(D.__mro__)
    # (<class '__main__.D'>, <class '__main__.B'>, <class '__main__.C'>, <class '__main__.A'>, <class 'object'>)

if __name__=='__main__':
    # test1()
    test2()