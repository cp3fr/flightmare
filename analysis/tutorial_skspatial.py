from skspatial.objects import Vector, Points, Line

vector = Vector([2, 0, 0])
print(vector.size)
print(vector.norm())
print(vector.unit())
print(vector.sum())
print(Vector([1, 0]).cosine_similarity([0, 1]).round(3))
print('')

points = Points([[1,2,3], [4,5,6], [7,8,8]])
print(points.are_collinear())
print(points.are_collinear(tol=1))
print('')

points = Points([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(points.are_collinear())
print('')

line = Line(point=[0,0,0], direction=[1,1,0])
print(line.project_point([5,6,7]))