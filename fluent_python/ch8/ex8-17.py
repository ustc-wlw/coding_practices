'''
弱引用
'''
import weakref

a_set = {0,1}

wref = weakref.ref(a_set)
print('if wref is None? ', wref is None)

print(wref, wref()) # <weakref at 0x0000024BC3DC46D8; to 'set' at 0x0000024BC3E17128> {0, 1}

a_set = {3,4}

print(wref, wref() is None) # <weakref at 0x0000024BC3DC46D8; dead> True

