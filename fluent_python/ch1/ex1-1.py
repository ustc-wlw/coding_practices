import collections
from random import choice

Card = collections.namedtuple('card',['rank', 'suit'])
suit_values = dict(spades=3, hearts=2, diamonds=1, clubs=0)

class FrenchDeck:
    ranks = [str(n) for n in range(2,11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self) -> None:
        self.cards = [Card(rank, suit) for suit in self.suits
                                        for rank in self.ranks]
        
    def __len__(self):
        return len(self.cards)

    def __getitem__(self, index):
        # assert index >= 0 and index < len(self.cards) - 1
        return self.cards[index]

deck = FrenchDeck()

def test1():
    deck = FrenchDeck()
    print('deck length: ', len(deck))
    print(deck[0], deck[-1])

    print('random choice deck is ', choice(deck))

    print(deck[:3])

    print('rank is A cards:')
    print(deck[12::13])

def test2():
    deck = FrenchDeck()
    # for card in deck:
    #     print(card)

    for card in reversed(deck):
        print(card)

def test3():
    b1 = Card('Q', 'hearts') in deck       
    b2 = Card('7', 'hello') in deck
    print(b1, b2)

def spades_high(card):
    rank_value = FrenchDeck.ranks.index(card.rank)
    return rank_value * len(suit_values) + suit_values[card.suit]

def test_sort():
    for i, card in enumerate(sorted(deck, key=spades_high)):
        print(i, card)

if __name__=='__main__':
    # test1()
    # test2()
    # test3()

    test_sort()