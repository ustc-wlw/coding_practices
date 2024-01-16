from array import array
import math
import reprlib
import numbers
import functools
import operator
import itertools

class Vector:
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
        # return tuple(self) == tuple(__o)
        return len(self) == len(__o) and all(a == b for a,b in zip(self, __o))

    def __abs__(self):
        return math.sqrt(sum(x**2 for x in self._components))

    def __bool__(self):
        return bool(abs(self))

    def __hash__(self) -> int:
        hashes = (hash(x) for x in self._components)
        ret = functools.reduce(operator.xor, hashes, 0)
        return ret

    def angle(self, n):
        r = math.sqrt(sum(x*x for x in self[n:]))
        a = math.atan2(r, self[n-1])
        if (n == len(self) - 1) and (self[-1] < 0):
            return math.pi * 2 - a
        else:
            return a

    def angles(self):
        return (self.angle(n) for n in range(1, len(self)))

    def __format__(self, __format_spec: str='') -> str:
        if __format_spec.endswith('h'):
            __format_spec = __format_spec[:-1]
            outer_fmt = '<{}>'
            coords = itertools.chain([abs(self)], self.angles())
        else:
            coords = self
            outer_fmt = '({})'
        components = (format(x, __format_spec) for x in coords)
        return outer_fmt.format(', '.join(components))

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
    print(format(Vector([-1,-1,-1,-1]), 'h'))

    print(format(Vector([2,2,2,2]), '.3eh'))

    print(format(Vector([0,1,0,0]), '0.5fh'))

if __name__=='__main__':
    test_vector()
    