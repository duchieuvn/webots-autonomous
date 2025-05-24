import pygame
import numpy as np
import cv2

MAP_SIZE = 1000  # 1000cm = 10m
window_size = (500, 500)

class Visualizer():
    def __init__(self):
        self.window_size = (500, 500)
        self.screen = self.init_pygame()
        pass
    
    def init_pygame(self):
        pygame.init()
        screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Occupancy Grid Map")
        return screen
    
    def clear_screen(self):
        self.screen.fill((0,0,0))

    def display_screen(self):
        pygame.display.flip()

    def convert_map_to_screen(self, map_x, map_y):
        screen_x = int(map_x * self.window_size[0] / MAP_SIZE)
        screen_y = int(map_y * self.window_size[1] / MAP_SIZE)
        return screen_x, screen_y

    def draw_robot(self, map_position):
        screen_x, screen_y = self.convert_map_to_screen(map_position[0], map_position[1])
        pygame.draw.circle(self.screen, (0, 0, 255), (screen_x, screen_y), 5)  # Bán kính = 5 pixel


    def create_map_surface(self, visual_map):
        map_resized = cv2.resize(visual_map, window_size)
        map_surface = pygame.surfarray.make_surface(map_resized)
        return map_surface

    def draw_path(self, path):
        scale_x = window_size[0] / MAP_SIZE
        scale_y = window_size[1] / MAP_SIZE
        scaled_path = [(int(x * scale_x), int(y * scale_y)) for (x, y) in path]

        for point in scaled_path:
            pygame.draw.circle(self.screen, (255, 0, 0), point, 1)
    
    def draw_robot_to_target(self, robot, target):
        current_pos = robot.get_map_position()
        target_pos = target
        scale_x = window_size[0] / MAP_SIZE
        scale_y = window_size[1] / MAP_SIZE
        p1 = (current_pos[0] * scale_x, current_pos[1] * scale_y)
        p2 = (target_pos[0] * scale_x, target_pos[1] * scale_y)
        self.draw_infinite_line((255, 0, 0), p1, p2, width=2)

    def draw_robot_orientation(self, robot):
        current_pos = robot.get_map_position()
        scale_x = window_size[0] / MAP_SIZE
        scale_y = window_size[1] / MAP_SIZE
        p1 = (current_pos[0] * scale_x, current_pos[1] * scale_y)
        heading_rad = robot.get_heading('rad')
        dir_vector = (np.cos(heading_rad), np.sin(heading_rad))
        p2 = (p1[0] + dir_vector[0] * 100, p1[1] - dir_vector[1] * 100)
        self.draw_infinite_line((0, 255, 0), p1, p2, width=2)

    def draw_line(self, map_p1, map_p2):
        screen_p1 = self.convert_map_to_screen(map_p1[0], map_p1[1])
        screen_p2 = self.convert_map_to_screen(map_p2[0], map_p2[1])
        pygame.draw.line(self.screen, (0, 100, 100), screen_p1, screen_p2, 1)

    def draw_infinite_line(self, color, p1, p2, width=2):
        x1, y1 = p1
        x2, y2 = p2
        w, h = self.screen.get_size()

        if x1 == x2:
            pygame.draw.line(self.screen, color, (x1, 0), (x1, h), width)
            return
        else:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1

            points = []

            y_at_left = b
            if 0 <= y_at_left <= h:
                points.append((0, int(y_at_left)))

            y_at_right = m * w + b
            if 0 <= y_at_right <= h:
                points.append((w, int(y_at_right)))

            if m != 0:
                x_at_top = -b / m
                if 0 <= x_at_top <= w:
                    points.append((int(x_at_top), 0))

                x_at_bottom = (h - b) / m
                if 0 <= x_at_bottom <= w:
                    points.append((int(x_at_bottom), h))

            if len(points) >= 2:
                pygame.draw.line(self.screen, color, points[0], points[1], width)

    def visualize_game(self, robot, path, target):
        self.clear_screen()        
        self.draw_path(path)
        self.draw_robot_to_target(robot, target)
        self.draw_robot_orientation(robot)
        pygame.display.flip()
    
    def update_screen_with_map(self, grid_map):
        visual_map = np.stack([grid_map*100] * 3, axis=-1)
        visual_map_rgb = np.transpose(visual_map, (1, 0, 2))
        map_surface = self.create_map_surface(visual_map_rgb)
        self.screen.blit(map_surface, (0, 0))
