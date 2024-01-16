'''
使用协程返回值
'''
from collections import namedtuple

Result = namedtuple('Result', 'count avg_value')

def averager():
    count = 0
    total = 0.0
    while True:
        val = yield
        if val is None:
            break
        count += 1
        total += val
        avg = total / count
    return Result(count, avg)

avg = averager()
next(avg)

avg.send(10)
avg.send(20)
avg.send(30)
try:
    avg.send(None)
except StopIteration as exc:
    result = exc.value
    print('result is: ', result)
else:
    print('else runing.....')