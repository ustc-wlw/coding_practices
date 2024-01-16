from inspect import signature

def clip(text:str, max_len:int=80) -> str:
    '''
    在max_len前面或后面的第一个空格截断文本
    '''
    end = None
    if len(text) > max_len:
        space_before = text.rfind(' ', 0,max_len)
        if space_before >= 0:
            end = space_before
        else:
            space_after = text.find(' ', max_len)
            if space_after >= 0:
                end = space_after
    if end is None:
        end = len(text)
    return text[:end].rstrip()

def test_clip():
    s = 'AB CD'
    print(clip(s, 7))

def test_inspect():
    print('__defaults__', clip.__defaults__)
    print('__code__', clip.__code__.co_argcount, clip.__code__.co_varnames)
    print('__annotations__', clip.__annotations__)

def test_inspect2():
    sig = signature(clip)
    print(sig)

    for name, param in sig.parameters.items():
        print(param.kind, ' : ', name, '=', param.default)

if __name__=='__main__':
    # test_clip()
    # test_inspect2()
    test_inspect()