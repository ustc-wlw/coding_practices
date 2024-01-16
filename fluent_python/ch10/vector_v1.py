from array import array
import math
import reprlib
import numbers

class Vector:
    ## for performance
    # __slots__ = ('__x', '__y')

    typecode = 'd'
    shortcut_names = 'xyzt'

    def __init__(self, components) -> None:
        self._components = array(self.typecode, components)

    @classmethod
    def frombytes(cls, octets):
        typecode = chr(octets[0])
        memv = memoryview(octets[1:]).cast(typecode)
        return cls(memv)

    def __iter__(self):
        return iter(self._components)

    def __repr__(self) -> str:
        components = reprlib.repr(self._components)
        components = components[components.find('['):-1]
        return 'Vector({})'.format(components)

    def __str__(self) -> str:
        return str(tuple(self))

    def __bytes__(self):
        return (bytes([ord(self.typecode)]) + 
                bytes(self._components))

    def __eq__(self, __o: object) -> bool:
        return tuple(self) == tuple(__o)

    def __abs__(self):
        return math.sqrt(sum(x**2 for x in self._components))

    def __bool__(self):
        return bool(abs(self))

    def __hash__(self) -> int:
        pass

    '''
    实现序列协议: __len__, __getitem__
    '''
    def __len__(self):
        return len(self._components)

    def __getitem__(self, index):
        cls = type(self)
        if isinstance(index, slice):
            return cls(self._components[index])
        elif isinstance(index, numbers.Integral):
            return self._components[index]
        else:
            msg = '{cls.__name__} indices must be integers'
            raise TypeError(msg.format(cls=cls))

    def __getattr__(self, __name: str):
        # print('-------- call getattribute func -------')
        cls = type(self)
        if len(__name) == 1:
            pos = cls.shortcut_names.find(__name)
            if pos >=0 and pos < len(self._components):
                return self._components[pos]
        msg = '{.__name__!r} object has not attribute {!r}'
        raise AttributeError(msg.format(cls, __name))

    def __setattr__(self, __name: str, __value: any) -> None:
        print('===== call setattr ===========')
        cls = type(self)
        if len(__name) == 1:
            if __name in cls.shortcut_names:
                msg = 'read only attribute {attribute!r}'
            elif __name.islower():
                msg = "can not set attributes 'a' to 'z' in {cls_name!r}"
            else:
                msg = ''
            if msg:
                raise AttributeError(msg.format(cls_name=cls.__name__,attribute=__name))
        super().__setattr__(__name, __value)
            
def test_vector():
    v1 = Vector([3,4])
    print(v1)

    octests = bytes(v1)
    print('v1 bytes: ', octests)

    print('abs for v1: ', abs(v1))

    print('bool for Vector2d: ', bool(v1), bool(Vector([0,0])))

    v2 = Vector(range(10))
    print(v2)

    print('length of v2: ', len(v2))
    print(v2[0], v2[-1], v2[1:4])

def test_vector2():
    v1 = Vector(range(10))
    print(v1.x, v1.y, v1.z, v1.t)
    print(v1.a)

def test_vector3():
    v1 = Vector(range(5))
    print(v1)
    print(v1.x)
    print('origin __dict__: ', v1.__dict__)

    print('*' * 40)

    v1.A = 100
    print(v1.x)
    print('v1 after updated: ', v1)
    print('new __dict__: ', v1.__dict__)

if __name__=='__main__':
    # test_vector()
    # test_vector2()
    test_vector3()
    