'''
协程实现求移动平均值
'''
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
print(avg)

print('pre avg next: ', next(avg))

print('1 avg: ', avg.send(10))
print('2 avg: ', avg.send(20))
print('3 avg: ', avg.send(30))

