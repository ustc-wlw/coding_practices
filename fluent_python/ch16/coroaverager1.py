from coroutil import coroutine

@coroutine
def averager():
    total = 0.0
    count = 0
    average = None
    while True:
        val = yield average
        total += val
        count += 1
        average = total / count

avg = averager()

print('1 send and avg value: ', avg.send(10))

print('2 send hello and avg value: ', avg.send('hello'))