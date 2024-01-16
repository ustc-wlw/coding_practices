from functools import reduce
from operator import mul

def fact(n):
    return reduce(lambda a,b : a*b, range(1,n+1))

def fact2(n):
    return reduce(mul, range(1, n+1))

print(fact(3))
print(fact2(3))