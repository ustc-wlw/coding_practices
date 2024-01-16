'''
构造副本, 元素是浅拷贝
'''
l1 = [3, [55,44], (7,8,9)]
l2 = list(l1)
def test1():
    print('id for l1 and l2: ', id(l1), id(l2))
    print('if l1 == l2? ', l1 == l2)
    print('l1 is l2? ', l1 is l2)

def test2():
    l1.append(100)
    print('after append 100 for l1, l1 & l2: ', l1, l2)

    l1[1].remove(55)
    print('after remove 55, l1 and l2: ', l1, l2)

    l2[1] += [33,22]
    l2[2] += (10,11)
    print('l1 l2: ', l1, l2)

if __name__=='__main__':
    # test1()
    test2()