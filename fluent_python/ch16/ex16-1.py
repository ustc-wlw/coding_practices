def simple_coroutine():
    print('-> coroutine stated ....')
    x = yield
    print('received data is {!r}'.format(x))

gen = simple_coroutine()
print(gen)

gen.send(20)

next(gen)

gen.send(10)