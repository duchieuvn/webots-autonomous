import pygame
from controller import Robot, Keyboard, Supervisor
from setup import setup_robot
from my_robot import MyRobot
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2

TIME_STEP = 32
MAX_VELOCITY = 26 
# v = w * r => v_max = 26 * 0.043 = 1.118 m/s
WHEEL_RADIUS = 0.043  # meters
AXLE_LENGTH = 0.18    # meters (distance between left and right wheels)

OBSTACLE_VALUE = 1
PATH_VALUE = 255
PIXEL_DETECT_RADIUS = 40

MAP_SIZE = 1000 # 1000cm = 10m
RESOLUTION = 0.01 # 1cm for 1 grid cell
grid_map = cv2.imread('../../textures/path_test_map.bmp', cv2.IMREAD_GRAYSCALE)
visual_map = cv2.imread('../../textures/map1.png', cv2.IMREAD_COLOR)
visual_map = cv2.cvtColor(visual_map, cv2.COLOR_BGR2RGB)


def init_pygame(window_size):
    pygame.init()
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Occupancy Grid Map")
    return screen

def create_map_surface(visual_map, window_size):
    map_resized = cv2.resize(visual_map, window_size)
    map_surface = pygame.surfarray.make_surface(np.transpose(map_resized, (1, 0, 2)))
    return map_surface

def generate_random_curved_path(start_point, end_point):
    step_size=50
    deviation=5
    
    x0, y0 = start_point
    x1, y1 = end_point

    dx = x1 - x0
    dy = y1 - y0
    distance = np.hypot(dx, dy)

    if distance == 0:
        return [start_point]

    num_steps = max(1, int(distance // step_size))

    points = []
    for i in range(num_steps + 1):
        t = i / num_steps
        x = x0 + t * dx
        y = y0 + t * dy

        # Thêm random nhỏ vào mỗi bước để tạo cong
        offset_x = random.randint(-deviation, deviation)
        offset_y = random.randint(-deviation, deviation)

        x += offset_x
        y += offset_y

        points.append((int(round(x)), int(round(y))))

    # Lọc duplicate points
    filtered_points = []
    seen = set()
    for p in points:
        if p not in seen:
            filtered_points.append(p)
            seen.add(p)

    # Đảm bảo điểm cuối đúng end_point
    if filtered_points[-1] != (int(round(x1)), int(round(y1))):
        filtered_points[-1] = (int(round(x1)), int(round(y1)))

    return filtered_points

def generate_path():
    start_point = [200, 250]
    path = []
    end_x = 400
    end_y = 400
    for i in range(3):
        random_int = random.randint(200, 350)
        if i % 2 == 0:
            end_x = start_point[0] + random_int
        else:
            end_y = start_point[1] + random_int

        path +=  generate_random_curved_path(start_point, (end_x, end_y))
        start_point = [end_x + 30, end_y+ 30]

    return path

def draw_path(path):
    # Scale path theo kích thước màn hình
    scale_x = window_size[0] / MAP_SIZE
    scale_y = window_size[1] / MAP_SIZE
    scaled_path = [(int(x * scale_x), int(y * scale_y)) for (x, y) in path]

    # Vẽ từng điểm
    for point in scaled_path:
        pygame.draw.circle(screen, (255, 0, 0), point, 1)  # chấm đỏ, bán kính 2 pixel

def draw_robot_to_target(screen, robot, target, window_size):
    current_pos = robot.get_map_position()
    target_pos = target
    scale_x = window_size[0] / MAP_SIZE
    scale_y = window_size[1] / MAP_SIZE
    p1 = (current_pos[0] * scale_x, current_pos[1] * scale_y)
    p2 = (target_pos[0] * scale_x, target_pos[1] * scale_y)
    draw_infinite_line(screen, (255, 0, 0), p1, p2, width=2)

def draw_robot_orientation(screen, robot, window_size):
    current_pos = robot.get_map_position()
    scale_x = window_size[0] / MAP_SIZE
    scale_y = window_size[1] / MAP_SIZE
    p1 = (current_pos[0] * scale_x, current_pos[1] * scale_y)

    heading_rad = robot.get_heading('rad')
    dir_vector = (np.cos(heading_rad), np.sin(heading_rad))
    p2 = (p1[0] + dir_vector[0] * 100, p1[1] - dir_vector[1] * 100)

    draw_infinite_line(screen, (0, 255, 0), p1, p2, width=2)

def draw_infinite_line(screen, color, p1, p2, width=2):
    x1, y1 = p1
    x2, y2 = p2
    w, h = screen.get_size()

    if x1 == x2:
        pygame.draw.line(screen, color, (x1, 0), (x1, h), width)
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
            pygame.draw.line(screen, color, points[0], points[1], width)

window_size = (500, 500)
screen = init_pygame(window_size)

def main():
    robot = MyRobot()
    try:
        window_size = (500, 500)
        screen = init_pygame(window_size)

        running = True
        path = generate_path()
        target_index = 0
        target = path[target_index]
        while robot.step(TIME_STEP) != -1 and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            screen.fill((0, 0, 0))
            robot.draw_position_in_map(visual_map)

            draw_path(path)
            draw_robot_to_target(screen, robot, target, window_size)
            draw_robot_orientation(screen, robot, window_size)

            pygame.display.flip()

            if robot.dwa_plan(target):
                target_index += 1
                if target_index < len(path):
                    target = path[target_index]
                else:
                    break
            # time.sleep(0.1)

        pygame.quit()

    except Exception as e:
        print("An error occurred:", e)
        pygame.quit()

main()