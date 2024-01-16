from vector2d_vo import Vector2d

def test1():
    brl = 1/2.43
    print(brl)

    print(format(brl, '0.4f'))

    format_str = '1 BRL = {rate:0.2f} USD'.format(rate=brl)
    print(format_str)

def test2():
    print(format(42, 'b')) ## byte
    print(format(42, 'x'))
    print(format(2/3, '.1%'))

def test3():
    v1 = Vector2d(3,4)
    print(format(v1)) ## call object class: str(self)
    print(format(v1, '.3f'))
    
def test4():
    v1 = Vector2d(1,1)
    print(format(v1, 'p'))
    print(format(v1, '.3ep'))
    print(format(v1, '0.5fp'))

def test5():
    v1 = Vector2d(3,4)
    print('hash value: ', hash(v1)) # TypeError: unhashable type: 'Vector2d'

    s1 = set([v1]) # TypeError: unhashable type: 'Vector2d'
    print(s1)

    # v1.x = 7

if __name__=='__main__':
    # test1()
    # test2()
    # test3()
    # test4()
    test5()