try:
    from analysis.functions.visattention import *
except:
    from functions.visattention import *


objWallCollider = wall_colliders(dims=(66, 36, 9), center=(0, 0, 4.5))
o = objWallCollider[0]

p0 = np.array([-32, -12, 3])
p1 = np.array([-32, -12, -3])

print(o.intersect(p0, p1))