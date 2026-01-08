from models import Environment
from GRN import GRN
import pygame as pg
from typing import Tuple, Dict, Any, List
import numpy as np
import multiprocessing as mp
from grn_visualization import run_visualizer

class Visualization:
    def __init__(self, environment: Environment, dt: float = 0.1):
        self.environment = environment
        self.show_sensors = False
        self.running = False
        self.scale = 1
        self.dt = dt
        pg.init()
        pg.display.set_caption("Sperm Simulation")
        self.screen = pg.display.set_mode((int(self.environment.width * self.scale), int(self.environment.height * self.scale)))

        self._viz_proc: mp.Process | None = None
        self._viz_queue: mp.Queue | None = None
        self._viz_graph_sent = False
        self._viz_frame = 0

    def _env_to_screen(self, position: Tuple[float, float]) -> Tuple[int, int]:
        return int(position[0] * self.scale), int(position[1] * self.scale)

    def _make_grn_payload_full(self) -> Dict[str, Any]:
        grn = self.environment.sperm_cell._grn

        edges: List[tuple] = []
        for src_name, gene in grn.genes.items():
            for (dst_name, w) in gene.connections:
                if dst_name in grn.genes:
                    edges.append((src_name, dst_name, float(w)))

        return {
            "groups": {
                "inputs": list(grn.input_genes),
                "internal": list(grn.internal_genes),
                "outputs": list(grn.output_genes),
            },
            "edges": edges,
            "values": grn.get_all_values(),
        }


    def _start_grn_visualizer(self):
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        self._viz_queue = mp.Queue(maxsize=5)
        self._viz_proc = mp.Process(
            target=run_visualizer,
            args=(self._viz_queue,),
            daemon=True
        )
        self._viz_proc.start()

        try:
            self._viz_queue.put_nowait(self._make_grn_payload_full())
            self._viz_graph_sent = True
        except Exception:
            self._viz_graph_sent = False

    def _push_grn_update(self):
        if self._viz_queue is None:
            return

        if not self._viz_graph_sent:
            try:
                self._viz_queue.put_nowait(self._make_grn_payload_full())
                self._viz_graph_sent = True
            except Exception:
                return

        self._viz_frame += 1
        if self._viz_frame % 2 != 0:
            return
        
        payload = self._make_grn_payload_full()
        try:
            self._viz_queue.put_nowait(payload)
        except Exception:
            pass


    def _render(self):
        self.screen.fill((255, 255, 255))

        if self.show_sensors:
            for sensor in self.environment.sperm_cell.sensors:
                pg.draw.circle(self.screen, (255, 0, 0), self._env_to_screen(sensor.get_absolute_position(self.environment.sperm_cell)), int(2 * self.scale))
                pg.draw.polygon(self.screen, (255, 200, 200), [
                    self._env_to_screen(sensor.get_absolute_position(self.environment.sperm_cell)),
                    self._env_to_screen((
                        sensor.get_absolute_position(self.environment.sperm_cell)[0] + sensor.range * np.cos(sensor.get_absolute_position(self.environment.sperm_cell)[2] - sensor.fov / 2),
                        sensor.get_absolute_position(self.environment.sperm_cell)[1] + sensor.range * np.sin(sensor.get_absolute_position(self.environment.sperm_cell)[2] - sensor.fov / 2)
                    )),
                    self._env_to_screen((
                        sensor.get_absolute_position(self.environment.sperm_cell)[0] + sensor.range * np.cos(sensor.get_absolute_position(self.environment.sperm_cell)[2] + sensor.fov / 2),
                        sensor.get_absolute_position(self.environment.sperm_cell)[1] + sensor.range * np.sin(sensor.get_absolute_position(self.environment.sperm_cell)[2] + sensor.fov / 2)
                    ))
                ])
            
        pg.draw.circle(self.screen, (0, 255, 0), self._env_to_screen(self.environment.ovum.position), int(self.environment.ovum.radius * self.scale))

        pg.draw.circle(self.screen, (0, 0, 255), self._env_to_screen(self.environment.sperm_cell.position), int(self.environment.sperm_cell.radius * self.scale))
        pg.draw.line(self.screen, (0, 0, 255), self._env_to_screen(self.environment.sperm_cell.position), 
            self._env_to_screen((
                self.environment.sperm_cell.position[0] - (self.environment.sperm_cell.radius + 5) * np.cos(self.environment.sperm_cell.orientation),
                self.environment.sperm_cell.position[1] - (self.environment.sperm_cell.radius + 5) * np.sin(self.environment.sperm_cell.orientation))
        ), 2)
        
       
        for obstacle in self.environment.obstacles:
            pg.draw.circle(self.screen, (255, 165, 0), self._env_to_screen(obstacle.position), int(obstacle.radius * self.scale))
        
        for danger in self.environment.dangers:
            pg.draw.circle(self.screen, (255, 0, 0), self._env_to_screen(danger.position), int(danger.radius * self.scale))

        for wbc in self.environment.white_blood_cells:
            pg.draw.circle(self.screen, (255, 0, 0), self._env_to_screen(wbc.position), int(wbc.radius * self.scale))
        
        pg.display.flip()

    def _step(self, dt: float) -> bool:
        status = self.environment.update(dt=self.dt)
        if status != 'ongoing':
            print(f"Simulation ended with status: {status}")
            return False
        return True

    def run(self):
        self._start_grn_visualizer()
        i = 0

        while True:
            print(i)
            for event in pg.event.get():
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE) or (event.type == pg.KEYDOWN and event.key == pg.K_q):
                    pg.quit()
                    break
                if event.type == pg.KEYDOWN and event.key == pg.K_s:
                    self.show_sensors = not self.show_sensors
                if event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                    self.running = not self.running
                if event.type == pg.KEYDOWN and (event.key == pg.K_PLUS or event.key == pg.K_EQUALS):
                    self.scale += 0.1
                    self.screen = pg.display.set_mode((int(self.environment.width * self.scale), int(self.environment.height * self.scale)))
                if event.type == pg.KEYDOWN and event.key == pg.K_MINUS:
                    self.scale = max(0.1, self.scale - 0.1)
                    self.screen = pg.display.set_mode((int(self.environment.width * self.scale), int(self.environment.height * self.scale)))
                if event.type == pg.KEYDOWN and event.key == pg.K_KP_ENTER and not self.running:
                    i += 1
                    if not self._step(self.dt):
                        pg.quit()
                        break

            if self.running:
                i += 1
                if not self._step(self.dt):
                    pg.quit()
                    break

            self._render()
            self._push_grn_update()
