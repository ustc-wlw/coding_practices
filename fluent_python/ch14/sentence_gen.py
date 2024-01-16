'''
实现生成器函数实现迭代器  __next__方法
'''
import re
import reprlib
from collections import abc

WORD_RE = re.compile('\w+')

class Sentence:
    def __init__(self, text:str) -> None:
        self._text = text
        self._words = WORD_RE.findall(self._text)

    ## 可迭代对象，返回迭代器
    def __iter__(self):
        for word in self._words:
            yield word

    def __repr__(self):
        return 'Sentence(%s)' % reprlib.repr(self._text)

s = Sentence('Hello World')
print('s is Iterable? ', isinstance(s, abc.Iterable)) # True
gen = iter(s)
print(gen)
print('gen is iterable? ', isinstance(gen, abc.Iterable), isinstance(gen, abc.Iterator))
print(next(gen))
print(next(gen))
print(next(gen))
print(next(gen))