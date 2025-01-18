from math import inf, pi, sqrt, pow

class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

class Line:
    def __init__(self, point_1: Point, point_2: Point):
        self.point_1 = point_1
        self.point_2 = point_2
    
    @property
    def length(self) -> float:
        return UtilFunctions.get_distance(self.point_1, self.point_2)

    @property
    def slope(self) -> float:
        if self.point_1.x == self.point_2.x:
            return inf
        
        if self.point_1.y == self.point_2.y:
            return 0
        
        return (self.point_2.y - self.point_1.y) / (self.point_2.x - self.point_1.x)

class Circle:
    def __init__(self, center: Point, radius: int):
        self.center = center
        self.radius = radius

    @property
    def circumference(self) -> float:
        return 2 * pi * self.radius
    
    @property
    def area(self) -> float:
        return pi * pow(self.radius, 2)
    
class Polygon():
    def __init__(self, vertices: list[Point]):
        self.vertices = vertices

    @property
    def perimeter(self):
        res = 0
        for i in range(len(self.vertices)):
            res += UtilFunctions.get_distance(self.vertices[i], self.vertices[(i + 1) % len(self.vertices)])

        return res

class UtilFunctions:
    @staticmethod
    def get_distance(point_1: Point, point_2: Point) -> float:
        return sqrt(pow(point_2.x - point_1.x, 2) + pow(point_2.y - point_1.y, 2))
    
    @staticmethod
    def are_parallel(line_1: Line, line_2: Line) -> bool:
        return line_1.slope == line_2.slope
    
    @staticmethod
    def are_perpendicular(line_1: Line, line_2: Line) -> bool:
        slopes = [inf, 0] # vertical and horizon
        if line_1.slope in slopes:
            slopes.remove(line_1.slope)
        if line_2.slope in slopes:
            slopes.remove(line_2.slope)

        if len(slopes) <= 0:
            return True # vertical line and horizon line are perpendicular
        return line_1.slope * line_2.slope == -1
    
    @staticmethod
    def intersect(circle_1: Circle, circle_2: Circle) -> bool:
        return UtilFunctions.get_distance(circle_1.center, circle_2.center) <= (circle_1.radius + circle_2.radius)


