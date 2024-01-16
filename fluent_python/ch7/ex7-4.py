b = 6

def f1(a):
    global b
    print('a',a)
    print('b',b)
    b = 9

f1(3)
print('b', b)