from operator import itemgetter

metro_data = [('Toko', 'JP', 36.933, (1,2)),
                ('Delhi NCR', 'IN', 21.935, (3,4)),
                ('Mexico', 'MX', 20.142, (5,6)),
                ('Sao Paulo', 'BR', 19.649, (7,8))]

def test_itemgetter():
    for city in sorted(metro_data, key=itemgetter(1)):
        print(city)
    
    cc_name = itemgetter(1,0)
    for city in metro_data:
        print(cc_name(city))

if __name__=='__main__':
    test_itemgetter()