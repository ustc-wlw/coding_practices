from functools import partial
from operator import methodcaller
from operator import mul

def test_methodcaller():
    s = 'The time has come'

    upcase = methodcaller('upper')
    print(upcase(s))

    rep = methodcaller('replace', ' ', '**')
    print(rep(s))

def test_partial():
    triple = partial(mul, 3)
    print(triple(7))

    data = list(map(triple, range(5)))
    print(data)

if __name__=='__main__':
    # test_methodcaller()

    test_partial()