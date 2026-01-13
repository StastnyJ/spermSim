from typing import List, Tuple
import numpy as np
import math
import json

from GRN import GRN


def closest_point_on_circle(center: Tuple[float, float], radius: float, point: Tuple[float, float]) -> Tuple[float, float]:
    """
    Returns the (x, y) coordinates of the closest point on a circle
    with center (cx, cy) and radius r to the point (px, py).
    """
    cx, cy = center
    px, py = point

    dx = px - cx
    dy = py - cy

    distance = math.hypot(dx, dy)

    if distance == 0:
        return (cx + radius, cy)

    scale = radius / distance
    closest_x = cx + dx * scale
    closest_y = cy + dy * scale

    return closest_x, closest_y


def distance_between_points(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


class Element:
    def __init__(self, position: Tuple[float, float] = (0, 0), radius: float = 1.0):
        self.position: Tuple[float, float] = position
        self.radius: float = radius

    def is_colliding(self, other: 'Element') -> bool:
        dist_sq = (self.position[0] - other.position[0]) ** 2 + (self.position[1] - other.position[1]) ** 2
        radius_sum = self.radius + other.radius
        return dist_sq <= radius_sum ** 2


class Sperm(Element):
    def __init__(
            self, position: Tuple[float, float] = (0, 0), orientation: float = 0.0, radius: float = 1.0, grn: GRN = None,
            max_speed: float = 1.0, max_rotation: float = np.pi / 4, sensors_range: float = 10.0, energy_decay: float = 0.002, energy_recovery: float = 0.001):
        super().__init__(position, radius)
        self.orientation: float = orientation
        self._max_speed: float = max_speed
        self._max_rotation: float = max_rotation
        self._grn: GRN = grn
        self.sensors: List[Sensor] = [
            Sensor(fov=np.pi / 1.92, range=sensors_range, relative_position=(radius * np.cos(np.pi / 4), -radius * np.sin(np.pi / 4)), relative_orientation=-np.pi/4),
            Sensor(fov=np.pi / 1.92, range=sensors_range, relative_position=(radius * np.cos(np.pi / 4), radius * np.sin(np.pi / 4)), relative_orientation=np.pi / 4),
            Sensor(fov=np.pi / 1.92, range=sensors_range, relative_position=(-radius * np.cos(np.pi / 4), radius * np.sin(np.pi / 4)), relative_orientation=3 * np.pi / 4),
            Sensor(fov=np.pi / 1.92, range=sensors_range, relative_position=(-radius * np.cos(np.pi / 4), -radius * np.sin(np.pi / 4)), relative_orientation=-3*np.pi / 4),
        ]
        self.energy: float = 1.0
        self.energy_decay: float = energy_decay
        self.energy_recovery: float = energy_recovery

    def update(self, dt: float, inputs: dict):
        if not self._grn:
            return
        self._grn.set_input(inputs)
        self._grn.update()
        rotation = self._grn.get_output()['rotation']
        speed = self._grn.get_output()['speed']
        self.orientation += 2 *(rotation - 0.5) * dt * self._max_rotation
        self.position = (
            self.position[0] + speed * dt * np.cos(self.orientation) * self._max_speed,
            self.position[1] + speed * dt * np.sin(self.orientation) * self._max_speed
        )
        self.energy = max(self.energy - self.energy_decay * dt * (speed + rotation), 0.0)
        if self.energy > 0:
            self.energy = min(self.energy_recovery * dt + self.energy, 1.0)

class WhiteBloodCell(Element):
    def __init__(
            self, position: Tuple[float, float] = (0, 0), radius: float = 1.0, max_speed: float = 1.0, sensor_range: float = 100.0):
        super().__init__(position, radius)
        self._max_speed: float = max_speed
        self.sensor_range: float = sensor_range
        self.orientation: float = 0.0

    def update(self, dt: float, sperm: Sperm):
        # Zjistíme vzdálenost ke spermii
        dx, dy = sperm.position[0] - self.position[0], sperm.position[1] - self.position[1]
        distance = np.sqrt(dx ** 2 + dy ** 2)
        
        # Pokud je spermie mimo kruhový dosah, nehýbeme se
        if distance > self.sensor_range:
            return
        
        # Jinak se pohybujeme směrem ke spermii
        if distance > 0:
            dx, dy = self._max_speed * dx / distance, self._max_speed * dy / distance
            self.position = (
                self.position[0] + dx * dt,
                self.position[1] + dy * dt
            )
            self.orientation = np.arctan2(dy, dx)


class Sensor:
    def __init__(self, fov: float = np.pi / 4, range: float = 10.0, relative_position: Tuple[float, float] = (0, 0), relative_orientation: float = 0.0):
        self.fov: float = fov
        self.range: float = range
        self.relative_position: Tuple[float, float] = relative_position
        self.relative_orientation: float = relative_orientation
    
    def get_absolute_position(self, cell) -> Tuple[float, float, float]:
        abs_x = cell.position[0] + self.relative_position[0] * np.cos(cell.orientation) - self.relative_position[1] * np.sin(cell.orientation)
        abs_y = cell.position[1] + self.relative_position[0] * np.sin(cell.orientation) + self.relative_position[1] * np.cos(cell.orientation)
        abs_orientation = cell.orientation + self.relative_orientation
        return abs_x, abs_y, abs_orientation

    def is_in_range(self, cell, element: Element) -> bool:
        abs_x, abs_y, abs_orientation = self.get_absolute_position(cell)
        dx = element.position[0] - abs_x
        dy = element.position[1] - abs_y
        distance = np.sqrt(dx ** 2 + dy ** 2)
        if distance > self.range + element.radius:
            return False
        angle_to_element = np.arctan2(dy, dx)
        angle_diff = (angle_to_element - abs_orientation + np.pi) % (2 * np.pi) - np.pi
        return abs(angle_diff) <= self.fov / 2


class Environment:

    def __init__(self, height: float = 0, width: float = 0, ovum: Element = None, sperm_cell: Sperm = None,
                 white_blood_cells: List[WhiteBloodCell] = None, obstacles: List[Element] = None, dangers: List[Element] = None):
        
        self.height: float = height
        self.width: float = width
        self.ovum: Element = ovum
        self.sperm_cell: Sperm = sperm_cell
        self.white_blood_cells: List[WhiteBloodCell] = white_blood_cells if white_blood_cells is not None else []
        self.obstacles: List[Element] = obstacles if obstacles is not None else []
        self.dangers: List[Element] = dangers if dangers is not None else []
        # TODO flow fields

    @staticmethod
    def from_json(json_file: dict, sperm_grn: GRN) -> 'Environment':
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        sperm_cell = Sperm(
            grn=sperm_grn,
            position=tuple(data['start_position']),
            orientation=data['start_orientation'],
            max_rotation=0.1, 
            max_speed=3,     
            radius=10.0,     
            sensors_range= 200.0 
        )
        return Environment(
            height=data["height"],
            width=data["width"],
            sperm_cell=sperm_cell,
            ovum=Element(position=tuple(data["ovum_position"]), radius=data["ovum_radius"]),
            obstacles=[Element(position=(o["x"], o["y"]), radius=o["radius"]) for o in data["obstacles"]],
            dangers=[Element(position=(d["x"], d["y"]), radius=d["radius"]) for d in data["dangers"]],
            white_blood_cells=[WhiteBloodCell(position=(wbc["x"], wbc["y"]), radius=15, sensor_range=150.0) for wbc in data["white_cells"]],
            # TODO flow fields
        )

    def compute_cell_inputs(self, cell: Sperm) -> dict:
        inputs = {}
        for i, sensor in enumerate(cell.sensors):
            sensor_pos = sensor.get_absolute_position(cell)[:2]
            
            range = sensor.range
            sensor.range = np.sqrt(self.width ** 2 + self.height ** 2)
            if sensor.is_in_range(cell, self.ovum):
                inputs[f'chem_signal_{i}'] = 1.0 - distance_between_points(sensor_pos, closest_point_on_circle(self.ovum.position, self.ovum.radius, sensor_pos)) / sensor.range
            else:
                inputs[f'chem_signal_{i}'] = 0.0
            sensor.range = range

            obstacle_signal = 0.0
            for obstacle in self.obstacles:
                if sensor.is_in_range(cell, obstacle):
                    dist = distance_between_points(sensor_pos, closest_point_on_circle(obstacle.position, obstacle.radius, sensor_pos))
                    obstacle_signal = max(obstacle_signal, 1.0 - dist / sensor.range)
            inputs[f'obstacle_signal_{i}'] = obstacle_signal

            danger_signal = 0.0
            for danger in self.dangers + self.white_blood_cells:
                if sensor.is_in_range(cell, danger):
                    dist = distance_between_points(sensor_pos, closest_point_on_circle(danger.position, danger.radius, sensor_pos))
                    danger_signal = max(danger_signal, 1.0 - dist / sensor.range)
            inputs[f'distance_to_danger_{i}'] = danger_signal

            inputs[f'force_flag_{i}'] = 0.0 # TODO compute force field sensor values

        inputs[f'energy'] = cell.energy
        return inputs

    def update(self, dt: float) -> str:
        if self.sperm_cell:
            inputs = self.compute_cell_inputs(self.sperm_cell)
            self.sperm_cell.update(dt, inputs)

        for wbc in self.white_blood_cells:
            wbc.update(dt, self.sperm_cell)
        
        if self.ovum.is_colliding(self.sperm_cell):
            return 'ovum_reached'
        
        for danger in self.dangers + self.white_blood_cells:
            if self.sperm_cell and self.sperm_cell.is_colliding(danger):
                return 'sperm_destroyed'
        
        if self.sperm_cell and self.sperm_cell.energy <= 0:
            return 'out_of_energy'
        
        for obstacle in self.obstacles:
            if self.sperm_cell and self.sperm_cell.is_colliding(obstacle):
                self.sperm_cell.position = closest_point_on_circle(obstacle.position, obstacle.radius + self.sperm_cell.radius, self.sperm_cell.position)

            for wbc in self.white_blood_cells:
                if wbc.is_colliding(obstacle):
                    wbc.position = closest_point_on_circle(obstacle.position, obstacle.radius + wbc.radius, wbc.position)

        # TODO handle flow fields
    
        if self.sperm_cell.position[0] < self.sperm_cell.radius:
            self.sperm_cell.position = (self.sperm_cell.radius, self.sperm_cell.position[1])
        if self.sperm_cell.position[0] > self.width - self.sperm_cell.radius:
            self.sperm_cell.position = (self.width - self.sperm_cell.radius, self.sperm_cell.position[1])
        if self.sperm_cell.position[1] < self.sperm_cell.radius:
            self.sperm_cell.position = (self.sperm_cell.position[0], self.sperm_cell.radius)
        if self.sperm_cell.position[1] > self.height - self.sperm_cell.radius:
            self.sperm_cell.position = (self.sperm_cell.position[0], self.height - self.sperm_cell.radius)

        return 'ongoing'
