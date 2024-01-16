from collections import namedtuple
import inspect

Result = namedtuple('Result', ['count', 'avg'])

'''sub gernerator'''
def averager():
    count = 0
    total = 0.0
    avg = None
    while True:
        val = yield avg
        if val is None:
            break
        total += val
        count += 1
        avg = total / count
    return Result(count, avg)

'''gernerator'''
def grouper(results:dict, key:str):
    while True:
        ## yield from handle StopIterationException
        results[key] = yield from averager()

def report(results):
    for key, result in sorted(results.items()):
        group, unit = key.split(';')
        print('{:2} {:5} averaging {:.2f}{}'.format(
        result.count, group, result.avg, unit))

'''client'''
def main(data:dict):
    results = {}
    for key, values in data.items():
        grouper_gen = grouper(results, key)
        next(grouper_gen)
        for val in values:
            ## send client data to sub_generator directly
            cur_avg = grouper_gen.send(val)
            print('current avg value is ', cur_avg)
        print('grouper gen state: ', inspect.getgeneratorstate(grouper_gen))
        grouper_gen.send(None)
    print('*'*40)
    report(results)

data = {
'girls;kg':
[40.9, 38.5, 44.3, 42.2, 45.2, 41.7, 44.5, 38.0, 40.6, 44.5],
'girls;m':
[1.6, 1.51, 1.4, 1.3, 1.41, 1.39, 1.33, 1.46, 1.45, 1.43],
'boys;kg':
[39.0, 40.8, 43.2, 40.8, 43.1, 38.6, 41.4, 40.6, 36.3],
'boys;m':
[1.38, 1.5, 1.32, 1.25, 1.37, 1.48, 1.25, 1.49, 1.46],
}

if __name__ == '__main__':
    main(data)