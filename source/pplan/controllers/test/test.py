import pygame
from controller import Robot, Keyboard, Supervisor
from setup import setup_robot
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

class MyRobot:
    def __init__(self):
        self.robot, self.motors, self.wheel_sensors, self.imu, self.camera_rgb, self.camera_depth, self.lidar, self.distance_sensors = setup_robot()
        self.grid_map = grid_map
        self.wheel_radius = WHEEL_RADIUS
        self.axle_length = AXLE_LENGTH


    def step(self, timestep):
        return self.robot.step(timestep)

    def stop_motor(self):
        for motor in self.motors.values():
            motor.setVelocity(0)

    def set_robot_velocity(self, left_speed, right_speed):
        self.motors['fl'].setVelocity(left_speed)
        self.motors['rl'].setVelocity(left_speed)
        self.motors['fr'].setVelocity(right_speed)
        self.motors['rr'].setVelocity(right_speed)

    def velocity_to_wheel_speeds(self, v, w):
        '''Convert linear and angular velocity to left and right wheel speeds.'''
        v_left = v - (self.axle_length / 2.0) * w
        v_right = v + (self.axle_length / 2.0) * w
        left_speed = v_left / self.wheel_radius
        right_speed = v_right / self.wheel_radius

        return left_speed, right_speed

    def go_backward_millisecond(self, ms=200):
        self.set_robot_velocity(-4, -4)
        self.step(ms)

    def get_heading(self, type='deg'):
        """Get the angle between robot heading and X-axis in world coordinates."""
        orientation = self.robot.getSelf().getOrientation()
        dir_x = orientation[0]
        dir_y = orientation[3]
        angle_rad = np.arctan2(dir_y, dir_x)
        if type == 'rad':
            return angle_rad
        elif type == 'deg':
            angle_deg = np.degrees(angle_rad)
            return angle_deg

    def get_angle_diff(self, map_target, type='deg'):
        heading = self.get_heading('rad')
        map_x, map_y = self.get_map_position()
        dx = map_target[0] - map_x
        dy = map_target[1] - map_y
        target_angle = np.arctan2(dy, dx)
        angle_diff = abs(target_angle - heading)

        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff

        return angle_diff if type == 'rad' else np.degrees(angle_diff)

    def get_compass_heading(self):
        compass_values = self.imu['compass'].getValues()
        rad = np.arctan2(compass_values[1], compass_values[0])
        deg = (np.degrees(rad) + 360) % 360
        return deg

    def get_distances(self):
        return [sensor.getValue() for sensor in self.distance_sensors]

    def get_position(self):
        return np.array(self.robot.getSelf().getPosition()[:2])

    def get_map_position(self):
        x, y = self.get_position()
        map_x = MAP_SIZE // 2 + int(x / RESOLUTION)
        map_y = MAP_SIZE // 2 - int(np.ceil(y / RESOLUTION))
        return np.array([map_x, map_y])

    def convert_to_map_coordinates(self, x, y):
        map_x = MAP_SIZE // 2 + int(x / RESOLUTION)
        map_y = MAP_SIZE // 2 - int(np.ceil(y / RESOLUTION))
        return int(map_x), int(map_y)
    
    def convert_to_world_coordinates(self, map_x, map_y):
        x = (map_x - MAP_SIZE // 2) * RESOLUTION
        y = (MAP_SIZE // 2 - map_y) * RESOLUTION
        return float(x), float(y)

    def draw_position_in_map(self, visual_map):
        pos = self.get_position()
        map_x, map_y = self.convert_to_map_coordinates(pos[0], pos[1])
        for i in range(-3, 4):
            for j in range(-3, 4):
                visual_map[map_y + j][map_x + i] = 255

    def get_local_target(self):
        map_x, map_y = self.get_map_position()
        r = PIXEL_DETECT_RADIUS

        intersect_list = []
        for x_border in [-r, r]:
            for j in range(-r, r+1):
                point_x = map_x + x_border
                point_y = map_y + j
                if self.grid_map[point_y, point_x] == PATH_VALUE and self.get_angle_diff([point_x, point_y]) <= 90:
                    visual_map[point_y, point_x] = (0,0,255)
                    intersect_list.append(np.array([point_x, point_y]))

        for y_border in [-r, r]:
            for i in range(-r, r+1):
                point_x = map_x + i
                point_y = map_y + y_border
                if self.grid_map[point_y, point_x] == PATH_VALUE and self.get_angle_diff([point_x, point_y]) <= 90:
                    visual_map[point_y, point_x] = (0,0,255)
                    intersect_list.append(np.array([point_x, point_y]))

        if len(intersect_list) == 0:
            return None
        else:
            dist = [np.linalg.norm(intersect_point - np.array([map_x, map_y])) for intersect_point in intersect_list]
            print('found path-------')
            return intersect_list[np.argmin(dist)]

    def there_is_obstacle(self, map_target):
        if self.grid_map[map_target[1], map_target[0]] == OBSTACLE_VALUE:
            return True
        return False

    def dwa_planner(self, world_target):
        MAX_SPEED = MAX_VELOCITY * self.wheel_radius
        # r = v / w
        v_samples = [0.08, 0.1, 0.15, 0.2, 0.5]
        # w_samples = [0.2, 0.5, 1, 1.5]
        w_samples = [0, 2, -2, 4, -4, 4.5, -4.5]

        best_score = -float('inf')
        # min_error = float('inf')
        best_v = 0.0
        best_w = 0.0

        x, y = self.get_position()
        theta = self.get_heading('rad')
        current_distance = np.linalg.norm(world_target - np.array([x, y]))

        dt = TIME_STEP / 1000 # Convert TIME_STEP to seconds
        for v in v_samples:
            for w in w_samples:
                cx, cy, ct = x, y, theta
                worse_path = False
                # Predict the position in 5 TIME_STEP intervals
                for _ in range(0, 5):
                    cx += v * np.cos(ct) * dt
                    cy += v * np.sin(ct) * dt
                    ct += w * dt

                    predicted_map_x, predicted_map_y = self.convert_to_map_coordinates(cx, cy)
                    if self.there_is_obstacle([predicted_map_x, predicted_map_y]):
                        print(f"Obstacle detected at ({predicted_map_x}, {predicted_map_y})")
                        worse_path = True
                        break
                    
                    predicted_distance = np.linalg.norm(world_target - np.array([cx, cy]))
                    if predicted_distance > current_distance + 0.1:
                        print(f"Predicted distance is worse: {predicted_distance} > {current_distance}")
                        worse_path = True
                        break
                
                if worse_path:
                    continue
                
                # predicted_angle_to_target = np.arctan2(cy-world_target[1], cx-world_target[0])
                predicted_angle_to_target = np.arctan2(world_target[1]-cy, world_target[0]-cx)
                heading_error = get_angle_diff(predicted_angle_to_target, ct)

                heading_score = np.cos(heading_error)
                distance_score = 1 - (predicted_distance / 2)  # you can fix some max, e.g., 2m
                speed_score = v / MAX_SPEED
                score = 4.0 * heading_score + 2.0 * distance_score + speed_score

                if score > best_score:

                    # print(f"Predicted angle to target: {np.degrees(predicted_angle_to_target)}")
                    # print(f"Heading error: {np.degrees(heading_error)}")
                    # print(f"Distance: {current_distance} {predicted_distance}")
                    # print(f"New min error: {error}")
                    best_score = score
                    best_v = v
                    best_w = w

        # print('angle diff: ', heading_error)
        # print(f"v: {best_v}, w: {best_v}, best score: {best_score}") 
        return best_v, best_w

    def dwa_plan(self, map_target):
        if self.get_map_distance(map_target) < 2:
            self.stop_motor()
            return True

        world_target = self.convert_to_world_coordinates(map_target[0], map_target[1])
        v, w = self.dwa_planner(world_target)

        left_speed, right_speed = self.velocity_to_wheel_speeds(v, w)
        print("Left speed: ", left_speed, "Right speed: ", right_speed)

        self.set_robot_velocity(left_speed, right_speed)
        return False

    def get_map_distance(self, target):
        a = self.get_map_position()
        return np.linalg.norm(a - np.array(target))


def get_angle_diff(a, b):
        diff = a - b
        while diff > np.pi:
            diff -= 2*np.pi
        while diff < -np.pi:
            diff += 2*np.pi
        return diff


robot = MyRobot()

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

def init_pygame(window_size):
    pygame.init()
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Occupancy Grid Map")
    return screen

def create_map_surface(visual_map, window_size):
    map_resized = cv2.resize(visual_map, window_size)
    map_surface = pygame.surfarray.make_surface(np.transpose(map_resized, (1, 0, 2)))
    return map_surface

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

window_size = (500, 500)
screen = init_pygame(window_size)

def main():
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
            print(len(path),'----------------')
            screen.fill((0, 0, 0))
            robot.draw_position_in_map(visual_map)

            # map_surface = create_map_surface(visual_map, window_size)
            # screen.blit(map_surface, (0, 0))

            draw_path(path)
            draw_robot_to_target(screen, robot, target, window_size)
            draw_robot_orientation(screen, robot, window_size)

            pygame.display.flip()

            if robot.dwa_plan(target):
                target_index += 1
                if target_index < len(path):
                    target = path[target_index]
                else:
                    print("Reached the end of the path.")
                    break
            print('---------------------------')
            # time.sleep(0.1)

        pygame.quit()

    except Exception as e:
        print("An error occurred:", e)
        pygame.quit()

main()