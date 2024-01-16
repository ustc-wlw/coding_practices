import os
print(os.curdir)
os.chdir('c:/Workspace/develop/python/fluent_python/ch15')

with open('./info.txt', encoding='utf-8') as fp:
    src = fp.read(60)
    print('content: ', src)
    print(len(src))

print(fp)
print(fp.closed, fp.encoding)