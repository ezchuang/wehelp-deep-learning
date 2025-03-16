from geometry_classes import *
from game_classes import *

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
        turn = 10
        enemies = [
            Enemy("E1", Point(-10, 2), Vector(2, -1)),
            Enemy("E2", Point(-8, 0), Vector(3, 1)),
            Enemy("E3", Point(-9, -1), Vector(3, 0)),
        ]

        towers = [
            BasicTower("T1", Point(-3, 2)),
            BasicTower("T2", Point(-1, -2)),
            BasicTower("T3", Point(4, 2)),
            BasicTower("T4", Point(7, 0)),
            AdvancedTower("A1", Point(1, 1)),
            AdvancedTower("A2", Point(4, -3)),
        ]

        game_handler = GameHandler(turn, enemies, towers)
        game_handler.start_procedure()

        for enemy in game_handler.enemies:
            print(f'label: {enemy.name}, position: ({enemy.point.x}, {enemy.point.y}), life point: {enemy.hp}')

if __name__ == "__main__":
    task_handler = TaskHandler()
    task_handler.task_1()
    task_handler.task_2()