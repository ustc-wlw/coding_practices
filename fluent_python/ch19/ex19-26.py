'''
通过特性输出管理的属性
'''
class BlackKnight:
    def __init__(self) -> None:
        self.members = ['a arm', 'another arm',
                        'a leg', 'another leg']

        self.phrases = ['Hello', 'World', 'Python', 'C++']

    @property
    def member(self):
        print('next member is: ')
        return self.members[0]

    @member.deleter
    def member(self):
        text = 'BlackKnight (loses {})\n -- {}'
        print(text.format(self.members.pop(0), self.phrases.pop(0)))

knight = BlackKnight()
print(knight.member)
print(vars(knight), vars(BlackKnight))

del knight.member
del knight.member


print(dir(knight))

print('vars: ', vars())
print('dir: ', dir())
