class Averager():
    def __init__(self) -> None:
        self.series = []

    def __call__(self, new_value):
        self.series.append(new_value)
        return sum(self.series) / len(self.series)

def make_averager():
    series = []
    def averager(new_value):
        series.append(new_value)
        total = sum(series)
        return total / len(series)
    return averager

avg = make_averager()

def test1():
    print(avg(10))
    print(avg(11))
    print(avg(12))

def test2():
    print('avg type: ', type(avg))
    print(avg.__code__.co_argcount)
    print(avg.__code__.co_varnames)
    print(avg.__code__.co_freevars)
    print('closure: ', avg.__closure__)
    print('closure cell contents: ', avg.__closure__[0].cell_contents)

if __name__=='__main__':
    test1()
    test2()