# from sentence import Sentence, Foo
from collections import abc

from sentence_iter import Sentence

def test():
    sentence = Sentence('the time has come, the Walrus saied')
    print(sentence)
    for s in sentence:
        print(s)


    print(sentence[0])

    print(isinstance(sentence, abc.Iterable))
    print(isinstance(sentence, abc.Iterator))

def test2():
    from sentence import Foo
    foo = Foo()
    print(isinstance(foo, abc.Iterable))
    print(issubclass(Foo, abc.Iterator))

def test3():
    s = Sentence('the time has come, the Walrus saied')
    it = iter(s)
    print(it)
    while True:
        try:
            c = next(it)
            print('char: ', c)
        except StopIteration:
            del it
            break
    

if __name__=='__main__':
    # test()

    # test2()
    test3()