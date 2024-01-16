def simple_coroutine2(a):
    print('-> started: a = ', a)
    b = yield a
    
    print('receive b is ', b)
    c = yield a + b
    print('receive c is ', c)

gen = simple_coroutine2(10)
print(gen)

print('1 next(gen): ', next(gen)) # 10
print(' after send 20 to gen')
print(gen.send(20)) # b=20
print('-> after send 30 to gen')
print(gen.send(30))  # c=30
