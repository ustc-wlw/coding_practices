import contextlib
import sys

'''
使用contextlib.@contextmanager装饰生成器函数，实现上下文管理对象协议
'''

@contextlib.contextmanager
def looking_glass():
    origin_writer = sys.stdout.write
    sys.stdout.write = lambda text: origin_writer(text[::-1])
    yield 'This is __enter__ return value!!'
    sys.stdout.write = origin_writer
    print('context manager __exit__ success!')

with looking_glass() as what:
    print('Hello World~~')
    print('what inside: ', what)

print('what outerside: ', what)