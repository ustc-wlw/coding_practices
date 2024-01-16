'''
实现序列协议
'''
import re
import reprlib

RE_WORD=re.compile('\w+')

class Sentence:
    def __init__(self, text) -> None:
        self._text = text
        self._words = RE_WORD.findall(text)

    def __getitem__(self, index):
        return self._words[index]

    def __len__(self):
        return len(self._words)

    def __repr__(self) -> str:
        return 'Sentence(%s) ' % reprlib.repr(self._text)

class Foo():
    def __iter__(self):
        pass