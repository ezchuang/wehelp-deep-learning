from geometry_classes import *

class Vector:
    def __init__(self, dx: int, dy: int):
        self.dx = dx
        self.dy = dy

class Enemy:
    def __init__(self, name: str, init_point: Point, move_speed: Vector, hp: int = 10):
        self.name = name
        self.point = init_point
        self.move_speed = move_speed
        self.hp = hp
        self.alive = True

    def move(self):
        if not self.alive:
            return
        
        self.point.x += self.move_speed.dx
        self.point.y += self.move_speed.dy

    def take_damage(self, damage: int):
        if not self.alive:
            return
        
        self.hp -= damage

        if self.hp > 0:
            return
        
        self.hp = 0
        self.alive = False

class Tower:
    def __init__(self, name: str, point: Point, attack: int, range: int):
        self.name = name
        self.point = point
        self.attack = attack
        self.range = range

    def attack_enemies(self, enemies: list[Enemy]):
        for enemy in enemies:
            if enemy.alive and ((self.point.x - enemy.point.x) ** 2 + (self.point.y - enemy.point.y) ** 2 <= self.range ** 2):
                enemy.take_damage(self.attack)

class BasicTower(Tower):
    def __init__(self, name: str, point: Point, attack: int = 1, range: int = 2):
        self.name = name
        self.point = point
        self.attack = attack
        self.range = range

class AdvancedTower(Tower):
    def __init__(self, name: str, point: Point, attack: int = 2, range: int = 4):
        self.name = name
        self.point = point
        self.attack = attack
        self.range = range

class GameHandler:
    def __init__(self, turn: int, enemies: list[Enemy], towers: list[Tower]):
        self.turn = turn
        self.enemies = enemies
        self.towers = towers

    def start_procedure(self):
        for _ in range(self.turn):
            self.move_enemies()
            self.towers_attack()

    def move_enemies(self):
        for enemy in self.enemies:
            enemy.move()

    def towers_attack(self):
        for tower in self.towers:
            tower.attack_enemies(self.enemies)