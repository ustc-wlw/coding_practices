'''
自定义实例属性,覆盖类属性
'''
from vector2d_vo import Vector2d

def test1():
    v1 = Vector2d(1.1, 2.2)
    dumped = bytes(v1)
    print(dumped)
    print('bytes length: ', len(dumped))

    print('*' * 40)
    # define instance attribute
    v1.typecode = 'f'
    dumped = bytes(v1)
    print(dumped)
    print('bytes length: ', len(dumped))
    print('v1 typecode: ', v1.typecode, v1.__dict__)

if __name__=='__main__':
    test1()
