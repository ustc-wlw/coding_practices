import monkey

def fun_i_am_human(self):
    print('I am human')

print(f'{monkey.Me.func_who_am_i}')

monkey.Me.func_who_am_i = fun_i_am_human

print(f'{monkey.Me.func_who_am_i}')

obj = monkey.Me()

print(f'{hasattr(obj, "func_who_am_i")}')
print(f'{hasattr(obj, "fun_i_am_human")}')

obj.func_who_am_i()