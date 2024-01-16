from vector2d_vo import Vector2d

v1 = Vector2d(3,4)
# print(v1.__dict__) # {'_Vector2d__x': 3.0, '_Vector2d__y': 4.0}

print(v1.__slots__)

print(v1._Vector2d__x)