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
        
        self.dirty_rects: List[pg.Rect] = []
        self.prev_dirty_rects: List[pg.Rect] = []
        self.full_redraw = True 
        self.frame_counter = 0
        self.last_screen_size = (int(self.environment.width * self.scale), int(self.environment.height * self.scale))
        
        self._load_sperm_animation()
        self.animation_frame = 0
        self.animation_speed = 0.01

    def _env_to_screen(self, position: Tuple[float, float]) -> Tuple[int, int]:
        return int(position[0] * self.scale), int(position[1] * self.scale)
    
    def _load_sperm_animation(self):
        import os
        animation_path = os.path.join(os.path.dirname(__file__), 'assets', 'sperm_animation.png')
        
        try:
            sheet = pg.image.load(animation_path).convert_alpha()
            
            self.frame_size = 32
            self.num_frames = 8
            
            self.sperm_frames = []
            for i in range(self.num_frames):
                frame = pg.Surface((self.frame_size, self.frame_size), pg.SRCALPHA)
                frame.blit(sheet, (0, 0), (0, i * self.frame_size, self.frame_size, self.frame_size))
                self.sperm_frames.append(frame)
            
            print(f"Animation sheet načten: {self.num_frames} framů")
        except Exception as e:
            print(f"Nepodařilo se načíst animation sheet: {e}")
            self.sperm_frames = []
            self.frame_size = 32
            self.num_frames = 1

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


    def _add_dirty_rect(self, center: Tuple[int, int], radius: int, margin: int = 10):
        """Přidá dirty rectangle pro kruh s marginem."""
        x, y = center
        rect = pg.Rect(
            x - radius - margin,
            y - radius - margin,
            (radius + margin) * 2,
            (radius + margin) * 2
        )
        screen_rect = self.screen.get_rect()
        rect = rect.clip(screen_rect)
        if rect.width > 0 and rect.height > 0:
            self.dirty_rects.append(rect)

    def _add_dirty_line_rect(self, start: Tuple[int, int], end: Tuple[int, int], width: int = 2, margin: int = 10):
        """Přidá dirty rectangle pro čáru."""
        x1, y1 = start
        x2, y2 = end
        min_x = min(x1, x2) - width - margin
        min_y = min(y1, y2) - width - margin
        max_x = max(x1, x2) + width + margin
        max_y = max(y1, y2) + width + margin
        rect = pg.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
        screen_rect = self.screen.get_rect()
        rect = rect.clip(screen_rect)
        if rect.width > 0 and rect.height > 0:
            self.dirty_rects.append(rect)

    def _add_dirty_polygon_rect(self, points: List[Tuple[int, int]], margin: int = 10):
        """Přidá dirty rectangle pro polygon."""
        if not points:
            return
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x = min(xs) - margin
        min_y = min(ys) - margin
        max_x = max(xs) + margin
        max_y = max(ys) + margin
        rect = pg.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
        screen_rect = self.screen.get_rect()
        rect = rect.clip(screen_rect)
        if rect.width > 0 and rect.height > 0:
            self.dirty_rects.append(rect)

    def _render(self):
        self.frame_counter += 1
        
        current_size = self.screen.get_size()
        if current_size != self.last_screen_size:
            self.full_redraw = True
            self.last_screen_size = current_size
        
        if self.frame_counter % 100 == 0:
            self.full_redraw = True
        
        was_full_redraw = self.full_redraw
        
        self.dirty_rects = []

        if self.full_redraw:
            self.screen.fill((255, 255, 255))
            self.full_redraw = False
            self.prev_dirty_rects = []
        else:
            for rect in self.prev_dirty_rects:
                self.screen.fill((255, 255, 255), rect)

        if self.show_sensors:
            for sensor in self.environment.sperm_cell.sensors:
                sensor_pos = self._env_to_screen(sensor.get_absolute_position(self.environment.sperm_cell))
                self._add_dirty_rect(sensor_pos, int(2 * self.scale))
                pg.draw.circle(self.screen, (255, 0, 0), sensor_pos, int(2 * self.scale))
                
                polygon_points = [
                    self._env_to_screen(sensor.get_absolute_position(self.environment.sperm_cell)),
                    self._env_to_screen((
                        sensor.get_absolute_position(self.environment.sperm_cell)[0] + sensor.range * np.cos(sensor.get_absolute_position(self.environment.sperm_cell)[2] - sensor.fov / 2),
                        sensor.get_absolute_position(self.environment.sperm_cell)[1] + sensor.range * np.sin(sensor.get_absolute_position(self.environment.sperm_cell)[2] - sensor.fov / 2)
                    )),
                    self._env_to_screen((
                        sensor.get_absolute_position(self.environment.sperm_cell)[0] + sensor.range * np.cos(sensor.get_absolute_position(self.environment.sperm_cell)[2] + sensor.fov / 2),
                        sensor.get_absolute_position(self.environment.sperm_cell)[1] + sensor.range * np.sin(sensor.get_absolute_position(self.environment.sperm_cell)[2] + sensor.fov / 2)
                    ))
                ]
                self._add_dirty_polygon_rect(polygon_points)
                pg.draw.polygon(self.screen, (255, 200, 200), polygon_points)
            
        ovum_pos = self._env_to_screen(self.environment.ovum.position)
        ovum_radius = int(self.environment.ovum.radius * self.scale)
        self._add_dirty_rect(ovum_pos, ovum_radius)
        pg.draw.circle(self.screen, (0, 255, 0), ovum_pos, ovum_radius)

        # Vykreslení spermie s animací
        sperm_pos = self._env_to_screen(self.environment.sperm_cell.position)
        sperm_radius = int(self.environment.sperm_cell.radius * self.scale)
        
        if self.sperm_frames and len(self.sperm_frames) > 0:
            speed = self.environment.sperm_cell._grn.get_output().get('speed', 0.0)
            adaptive_animation_speed = speed * 0.1
            
            self.animation_frame += adaptive_animation_speed
            if self.animation_frame >= self.num_frames:
                self.animation_frame = 0
            
            current_frame = self.sperm_frames[int(self.animation_frame)]
            
            scaled_size = int(self.frame_size * self.scale)
            scaled_frame = pg.transform.scale(current_frame, (scaled_size, scaled_size))
            
            angle_deg = -np.degrees(self.environment.sperm_cell.orientation)
            rotated_frame = pg.transform.rotate(scaled_frame, angle_deg)
            
            frame_rect = rotated_frame.get_rect(center=sperm_pos)
            
            self._add_dirty_rect(sperm_pos, max(scaled_size, sperm_radius) + 10)
            
            self.screen.blit(rotated_frame, frame_rect)
        else:
            self._add_dirty_rect(sperm_pos, sperm_radius)
            pg.draw.circle(self.screen, (0, 0, 255), sperm_pos, sperm_radius)
            
            line_end = self._env_to_screen((
                self.environment.sperm_cell.position[0] - (self.environment.sperm_cell.radius + 5) * np.cos(self.environment.sperm_cell.orientation),
                self.environment.sperm_cell.position[1] - (self.environment.sperm_cell.radius + 5) * np.sin(self.environment.sperm_cell.orientation))
            )
            self._add_dirty_line_rect(sperm_pos, line_end, 2)
            pg.draw.line(self.screen, (0, 0, 255), sperm_pos, line_end, 2)
        
       
        for obstacle in self.environment.obstacles:
            obs_pos = self._env_to_screen(obstacle.position)
            obs_radius = int(obstacle.radius * self.scale)
            self._add_dirty_rect(obs_pos, obs_radius)
            pg.draw.circle(self.screen, (255, 165, 0), obs_pos, obs_radius)
        
        for danger in self.environment.dangers:
            danger_pos = self._env_to_screen(danger.position)
            danger_radius = int(danger.radius * self.scale)
            self._add_dirty_rect(danger_pos, danger_radius)
            pg.draw.circle(self.screen, (255, 0, 0), danger_pos, danger_radius)

        for wbc in self.environment.white_blood_cells:
            wbc_pos = self._env_to_screen(wbc.position)
            wbc_radius = int(wbc.radius * self.scale)
            self._add_dirty_rect(wbc_pos, wbc_radius)
            pg.draw.circle(self.screen, (255, 0, 0), wbc_pos, wbc_radius)
        
        all_dirty_rects = self.prev_dirty_rects + self.dirty_rects
        
        if was_full_redraw:
            pg.display.flip()
        elif all_dirty_rects:
            pg.display.update(all_dirty_rects)
        else:
            pg.display.flip()
        
        self.prev_dirty_rects = self.dirty_rects[:]

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
                    self.full_redraw = True 
                if event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                    self.running = not self.running
                if event.type == pg.KEYDOWN and (event.key == pg.K_PLUS or event.key == pg.K_EQUALS or event.key == pg.K_KP_PLUS):
                    self.scale += 0.1
                    self.screen = pg.display.set_mode((int(self.environment.width * self.scale), int(self.environment.height * self.scale)))
                    self.last_screen_size = self.screen.get_size()
                    self.full_redraw = True
                if event.type == pg.KEYDOWN and (event.key == pg.K_MINUS or event.key == pg.K_KP_MINUS):
                    self.scale = max(0.1, self.scale - 0.1)
                    self.screen = pg.display.set_mode((int(self.environment.width * self.scale), int(self.environment.height * self.scale)))
                    self.last_screen_size = self.screen.get_size()
                    self.full_redraw = True
                if event.type == pg.KEYDOWN and (event.key == pg.K_KP_ENTER or event.key == pg.K_RETURN) and not self.running:
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
