class Gizmo:
    def __init__(self) -> None:
        print('Gizmo id %d' % id(self))

def test1():
    x = Gizmo()
    y = Gizmo() * 2
    print(dir())

if __name__=='__main__':
    test1()