from operator import attrgetter
from collections import namedtuple

metro_data = [('Toko', 'JP', 36.933, (1,2)),
                ('Delhi NCR', 'IN', 21.935, (3,4)),
                ('Mexico', 'MX', 20.142, (5,6)),
                ('Sao Paulo', 'BR', 19.649, (7,8))]

LatLong = namedtuple('LatLong', 'lat long')
Metropolis = namedtuple('Metropolis', 'name cc pop coord')
metro_areas = [Metropolis(name, cc, pop, LatLong(lat, long)) for name, cc, pop, (lat, long) in metro_data]
print(metro_areas[0])

print(metro_areas[0].coord.lat)

name_lat = attrgetter('name', 'coord.lat')
for city in metro_areas:
    print(name_lat(city))


import operator as op
print([name for name in dir(op) if not name.startswith('_')])