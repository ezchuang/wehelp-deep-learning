from geometry_classes import *

class TaskHandler:
    def __init__(self):
        pass

    def task_1(self):
        line_a = Line(Point(2, 4), Point(-6, 1))
        line_b = Line(Point(2, 2), Point(-6, -1))
        print("Are Line A and Line B parallel? ", UtilFunctions.are_parallel(line_a, line_b))

        line_c = Line(Point(-1, 6), Point(-4, -4))
        print("Are Line C and Line A perpendicular? ", UtilFunctions.are_perpendicular(line_a, line_c))

        circle_a = Circle(Point(6, 3), 2)
        print("Print the area of Circle A. ", circle_a.area)

        circle_b = Circle(Point(8, 1), 1)
        print("Do Circle A and Circle B intersect? ", UtilFunctions.intersect(circle_a, circle_b))

        polygon_a = Polygon([Point(2, 0), Point(-1, -2), Point(4, -4), Point(5, -1)])
        print("Print the perimeter of Polygon A. ", polygon_a.perimeter)

    def task_2(self):
        # enemies move
        # if the attack reaches its target
        # subtract life points
        # if the remaining life is 0
            # the enemy can no longer move
        pass

if __name__ == "__main__":
    task_handler = TaskHandler()
    task_handler.task_1()