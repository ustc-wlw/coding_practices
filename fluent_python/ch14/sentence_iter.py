
import re
import reprlib
import collections.abc

RE_WORD=re.compile('\w+')

class Sentence:
    def __init__(self, text) -> None:
        self._text = text
        self._words = RE_WORD.findall(self._text)

    ## 自身是可迭代对象，返回一个迭代器
    def __iter__(self):
        return SentenceIterator(self._words)

'''
抽象基类Iterator的子类, 实现__next__和 __iter__方法
collections.abc.Iterable [__iter__] --> collections.abc.Iterator [__next__] --> SentenceIterator
'''
class SentenceIterator:
    def __init__(self, words=None) -> None:
        self._words = words
        self._index = 0

    def __next__(self):
        try:
            val = self._words[self._index]
        except IndexError:
            raise StopIteration()
        else:
            self._index += 1
            return val
        
    def __iter__(self):
        return self

print(issubclass(SentenceIterator, collections.abc.Iterator))
print(isinstance(SentenceIterator(), collections.abc.Iterator))