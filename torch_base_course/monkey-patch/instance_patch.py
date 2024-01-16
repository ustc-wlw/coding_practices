import monkey
import types

m1 = monkey.Me()
m2 = monkey.Me()

def func_i_am_human(self):
    print(f'I am a human')

m2.func_who_am_i = types.MethodType(func_i_am_human, m2)

m1.func_who_am_i()
m2.func_who_am_i()