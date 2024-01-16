class A:
    def __init__(self) -> None:
        self.data_a = 'A class attribute'

class B(A):
    def __init__(self) -> None:
        super().__init__()
        self.data_b = 'B class attribute'

a = A()
b = B()
def test1():
    print(hasattr(b, 'data_b'))
    print(hasattr(b, 'data_a'))
    setattr(b, 'new_attr', 100)
    print('b dict: ', vars(b))

if __name__=='__main__':
    test1()