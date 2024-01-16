import sys

# print(sys.path)

from array import array
import math

class Vector2d:
    ## for performance
    # __slots__ = ('__x', '__y')

    typecode = 'd'

    def __init__(self, x, y) -> None:
        self.__x = float(x)
        self.__y = float(y)

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @classmethod
    def frombytes(cls, octets):
        typecode = chr(octets[0])
        memv = memoryview(octets[1:]).cast(typecode)
        return cls(*memv)

    def __iter__(self):
        return (e for e in (self.x, self.y))

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return '{}({!r}, {!r})'.format(class_name, *self)

    def __str__(self) -> str:
        # return '({!r}, {!r})'.format(*self)
        return str(tuple(self))

    def __bytes__(self):
        return (bytes([ord(self.typecode)]) + 
                bytes(array(self.typecode, self)))

    def __eq__(self, __o: object) -> bool:
        return tuple(self) == tuple(__o)

    def __abs__(self):
        return math.hypot(self.x, self.y)

    def __bool__(self):
        return bool(abs(self))

    def __hash__(self) -> int:
        return hash(self.x) ^ hash(self.y)

    def angle(self):
        return math.atan2(self.x, self.y)

    # def __format__(self, __format_spec: str='') -> str:
    #     components = (format(e, __format_spec) for e in self)
    #     return '({}, {})'.format(*components)
    def __format__(self, __format_spec: str='') -> str:
        if __format_spec.endswith('p'):
            __format_spec = __format_spec[:-1]
            coords = (abs(self), self.angle())
            outer_fmt = '<{}, {}>'
        else:
            coords = self
            # coords = (self.x, self.y)
            outer_fmt = '({}, {})'
        components = (format(e, __format_spec) for e in coords)
        return outer_fmt.format(*components)

def test_vector2d():
    v1 = Vector2d(3,4)
    print(v1.x, v1.y)

    x, y = v1
    print('* opration for vector2d: ', x,y)

    print('v1 is ', v1)

    v1_clone = eval(repr(v1))
    print('v1 == v1_clone? ', v1 == v1_clone)

    octests = bytes(v1)
    print('v1 bytes: ', octests)

    print('abs for v1: ', abs(v1))

    print('bool for Vector2d: ', bool(v1), bool(Vector2d(0,0)))




if __name__=='__main__':
    test_vector2d()
    