charles = {'name':'Charles L. Dodgson', 'born':1832}

lewis = charles

print(lewis is charles)

alex = {'name':'Charles L. Dodgson', 'born':1832}
print(alex == lewis)
print(alex is lewis)