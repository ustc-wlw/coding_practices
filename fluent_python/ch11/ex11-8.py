'''
定义抽象基类的子类
'''
import collections

Card = collections.namedtuple('Card','rank suit')

class FrenchDeck2(collections.MutableSequence):
    ranks = [str(n) for n in range(2,11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self) -> None:
        self._cards = [Card(rank, suit) for rank in self.ranks
                                        for suit in self.suits]

    def __len__(self) -> int:
        return len(self._cards)

    def __getitem__(self, index):
        return self._cards[index]

    def __setitem__(self, index, value):
        self._cards[index] = value

    def __delitem__(self, index):
        del self._cards[index]

    def insert(self, index: int, value: any) -> None:
        self._cards.insert(index, value)

f1 = FrenchDeck2()