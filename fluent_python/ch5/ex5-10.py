import inspect

def tag(name, *content, cls=None, **attrs):
    if cls:
        attrs['class']=cls
    if attrs:
        attr_str = ' '.join(' %s=%s' % (k,v) for (k,v) in sorted(attrs.items()))
    else:
        attr_str = ''
    if content:
        return '\n'.join('<%s%s>%s<%s>' % (name, attr_str, c, name)
                                            for c in content)
    else:
        return '<%s%s />' % (name, attr_str)

def test_tag():
    # print(tag('br'))
    # print(tag(name='img', content='testing'))
    print(tag('br', 'hello', 'world'))

def test_inspect():
    sig = inspect.signature(tag)
    print(sig)

    my_tag = {'name':'img', 'title':'Sunset Boulevard', 'src':'sunset', 'cls':'framed'}
    bound_args = sig.bind(**my_tag)
    for name, value in bound_args.arguments.items():
        print(name, '=', value)
    del my_tag['name']

    sig.bind(**my_tag)
    

if __name__=='__main__':
    # test_tag()
    test_inspect()