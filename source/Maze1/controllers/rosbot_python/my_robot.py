import numpy as np
from setup import setup_robot
import cv2
import random
from visualization import Visualizer
import pygame
import math

TIME_STEP = 32
MAX_VELOCITY = 26
WHEEL_RADIUS = 0.043
AXLE_LENGTH = 0.18
MAP_SIZE = 1000 # 10m x 10.0m grid map
# 1000 pixels = 10m, so each pixel is 0.01m = 1cm
RESOLUTION = 0.01

OBSTACLE_VALUE = 1
FREESPACE_VALUE = 0
UNKNOWN_VALUE = 255

grid_map = np.full((MAP_SIZE, MAP_SIZE), FREESPACE_VALUE, dtype=np.uint8)
obstacle_score_map = np.full((MAP_SIZE, MAP_SIZE), 2, dtype=np.float32)

def get_angle_diff(a, b):
        diff = a - b
        while diff > np.pi:
            diff -= 2*np.pi
        while diff < -np.pi:
            diff += 2*np.pi
        return diff

class MyRobot:
    def __init__(self):
        self.robot, self.motors, self.wheel_sensors, self.imu, self.camera_rgb, self.camera_depth, self.lidar, self.distance_sensors = setup_robot()
        self.time_step = TIME_STEP
        self.grid_map = grid_map
        self.obstacle_score_map = obstacle_score_map
        self.wheel_radius = WHEEL_RADIUS
        self.axle_length = AXLE_LENGTH

    def step(self, time_step=TIME_STEP):
        return self.robot.step(time_step)

    def is_turning(self):
        # Check if the robot is turning
        left_speed = self.motors['fl'].getVelocity()
        right_speed = self.motors['fr'].getVelocity()
        return abs(left_speed - right_speed) > 0.01

    def stop_motor(self):
        for motor in self.motors.values():
            motor.setVelocity(0)

    def set_robot_velocity(self, left_speed, right_speed):
        self.motors['fl'].setVelocity(left_speed)
        self.motors['rl'].setVelocity(left_speed)
        self.motors['fr'].setVelocity(right_speed)
        self.motors['rr'].setVelocity(right_speed)

    def velocity_to_wheel_speeds(self, v, w):
        v_left = v - (self.axle_length / 2.0) * w
        v_right = v + (self.axle_length / 2.0) * w
        left_speed = v_left / self.wheel_radius
        right_speed = v_right / self.wheel_radius
        return left_speed, right_speed

    def get_heading(self, type='deg'):
        # Calculate the angle from robot direction to the x-axis
        orientation = self.robot.getSelf().getOrientation()
        dir_x = orientation[0]
        dir_y = orientation[3]
        angle_rad = np.arctan2(dir_y, dir_x)
        if type == 'rad':
            return angle_rad
        elif type == 'deg':
            return np.degrees(angle_rad)

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

    def get_distances(self):
        return [sensor.getValue() for sensor in self.distance_sensors]

    def get_position(self):
        return np.array(self.robot.getSelf().getPosition()[:2])

    def get_map_position(self):
        x, y = self.get_position()
        map_x = MAP_SIZE // 2 + int(x / RESOLUTION)
        map_y = MAP_SIZE // 2 - int(np.ceil(y / RESOLUTION))
        return np.array([map_x, map_y])

    def get_map_distance(self, map_target):
        a = self.get_map_position()
        return np.linalg.norm(a - np.array(map_target))

    def convert_to_map_coordinates(self, x, y):
        map_x = MAP_SIZE // 2 + int(x / RESOLUTION)
        map_y = MAP_SIZE // 2 - int(np.ceil(y / RESOLUTION))
        return int(map_x), int(map_y)

    def convert_to_world_coordinates(self, map_x, map_y):
        x = (map_x - MAP_SIZE // 2) * RESOLUTION
        y = (MAP_SIZE // 2 - map_y) * RESOLUTION
        return float(x), float(y)

    def adapt_direction(self):
        def turn_right_milisecond(s=200):  
            self.set_robot_velocity(4, -4)
            self.step(s)

        def turn_left_milisecond(s=200):
            self.set_robot_velocity(-4, 4)
            self.step(s)

        def go_backward_milisecond(s=200):
            self.set_robot_velocity(-4, -4)
            self.step(s)

        # while there is an obstacle in front
        count = 0
        self.stop_motor()
        last_turn = 'right'
        distances = self.get_distances()
        while (min(distances[0], distances[2]) < 0.3 and count < 4):
            second = random.randint(100, 300)
            if distances[0] < distances[2]:
                turn_right_milisecond(second)
                last_turn = 'right'
            else:
                turn_left_milisecond(second)
                last_turn = 'left'

            count += 1

            distances = self.get_distances()
        
        if count == 4:
            # if there is an obstacle behind
            if (min(distances[1], distances[3]) > 0.3):
                go_backward_milisecond(200)
            
            if last_turn == 0:
                turn_left_milisecond(800)
            else:
                turn_right_milisecond(800)


    def dwa_planner(self, world_target):
        MAX_SPEED = MAX_VELOCITY * self.wheel_radius
        v_samples = [0.1, 0.2, 0.4, 0.5]
        w_samples = [0, 2, -2, 4.5, -4.5]

        best_score = -float('inf')
        best_v = 0.0
        best_w = 0.0

        x, y = self.get_position()
        theta = self.get_heading('rad')
        current_distance = np.linalg.norm(world_target - np.array([x, y]))

        # Predict the robot's future position after 5 TIME_STEP
        dt = TIME_STEP / 1000 # Convert to TIME_STEP to seconds
        for v in v_samples:
            for w in w_samples:
                cx, cy, ct = x, y, theta
                good_path = True
                # Calculate the robot position (cx, cy) after 5 steps
                for _ in range(0, 5):
                    cx += v * np.cos(ct) * dt
                    cy += v * np.sin(ct) * dt
                    ct += w * dt

                    predicted_map_x, predicted_map_y = self.convert_to_map_coordinates(cx, cy)
                    if self.there_is_obstacle([predicted_map_x, predicted_map_y]):
                        good_path = False
                        break
                    
                    # If the furtue distance from the target is greater than the current distance
                    predicted_distance = np.linalg.norm(world_target - np.array([cx, cy]))
                    if predicted_distance - current_distance > 0.2:
                        good_path = False
                        break

                if not good_path:
                    continue

                predicted_angle_to_target = np.arctan2(world_target[1]-cy, world_target[0]-cx)
                heading_error = get_angle_diff(predicted_angle_to_target, ct)

                heading_score = np.cos(heading_error)
                distance_score = 1 - (predicted_distance / 2)
                speed_score = v / MAX_SPEED
                score = 4.0 * heading_score + 3.5 * distance_score + speed_score

                if score > best_score:
                    best_score = score
                    best_v = v
                    best_w = w

        return best_v, best_w

    def follow_local_target(self, map_target):
        # Return True if the robot reached the target
        if self.get_map_distance(map_target) < 4:
            self.stop_motor()
            return True

        world_target = self.convert_to_world_coordinates(map_target[0], map_target[1])
        v, w = self.dwa_planner(world_target)
        left_speed, right_speed = self.velocity_to_wheel_speeds(v, w)
        self.set_robot_velocity(left_speed, right_speed)
        return False

    def there_is_obstacle(self, map_target):
        if self.grid_map[map_target[1], map_target[0]] == OBSTACLE_VALUE:
            return True
        return False
    
    def explore(self):
        vis = Visualizer()
        
        running = True
        count = 0
        while self.step(self.time_step) != -1 and count < 1000:
            for event in pygame.event.get(): 
                if event.type == pygame.QUIT:
                    running = False
            vis.clear_screen()

            self.adapt_direction()
            self.set_robot_velocity(8,8)
            points = self.get_pointcloud_world_coordinates()
            map_points = self.convert_to_map_coordinate_matrix(points)
            
            if count % 20 == 0 and not self.is_turning():
                # for map_point in map_points:
                    # self.draw_bresenham_line(map_point)
                    self.bresenham_to_obstacle_score(map_points)
                    self.update_grid_map()
                    # print(self.obstacle_score_map[100:200, 500:700])
                    # vis.draw_line(cur_map_pos, map_point)

            vis.update_screen_with_map(self.grid_map)
            vis.draw_robot(self.get_map_position())
            vis.display_screen()

            count += 1

        start_point = [200, 250]
        end_point = [600, 500]
        return self.grid_map, start_point, end_point 

    def visualize_grid_map(self, grid_map, window_name="Maze Grid Map"):
        visual_map = (1 - grid_map) * 255  # 0 -> 255 (white), 1 -> 0 (black)
        visual_map = visual_map.astype(np.uint8)
        resized_map = cv2.resize(visual_map, (500, 500), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(window_name, resized_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def generate_maze_grid_map(self, size=1000, wall_thickness =10, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        grid = np.zeros((size, size), dtype=np.uint8)

        def divide(x, y, w, h):
            if w < 2 * wall_thickness or h < 2 * wall_thickness:
                return

            horizontal = w < h
            if horizontal:
                if h - 2 * wall_thickness <= 0:
                    return
                wy = y + random.randrange(wall_thickness, h - wall_thickness)
                passage_x = x + random.randrange(0, w)
                for i in range(x, x + w):
                    if abs(i - passage_x) >= wall_thickness // 2:
                        grid[wy:wy + wall_thickness, i] = 1
                divide(x, y, w, wy - y)
                divide(x, wy + wall_thickness, w, y + h - wy - wall_thickness)
            else:
                if w - 2 * wall_thickness <= 0:
                    return
                wx = x + random.randrange(wall_thickness, w - wall_thickness)
                passage_y = y + random.randrange(0, h)
                for i in range(y, y + h):
                    if abs(i - passage_y) >= wall_thickness // 2:
                        grid[i, wx:wx + wall_thickness] = 1
                divide(x, y, wx - x, h)
                divide(wx + wall_thickness, y, x + w - wx - wall_thickness, h)

        # Draw outer walls
        grid[:wall_thickness, :] = 1
        grid[-wall_thickness:, :] = 1
        grid[:, :wall_thickness] = 1
        grid[:, -wall_thickness:] = 1

        # Start recursive division
        divide(wall_thickness, wall_thickness, size - 2 * wall_thickness, size - 2 * wall_thickness)

        return grid

    def find_path(self, start_point, end_point):
    
        def heuristic(a, b):
         # Octile distance approximation without math.sqrt (approx sqrt(2) = 1.4)
            dx = abs(a[0] - b[0])
            dy = abs(a[1] - b[1])
            D = 1
            D2 = 1.4
            return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

        def get_neighbors(pos):
            x, y = pos
            neighbors = []
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),           (0, 1),
                        (1, -1),  (1, 0),  (1, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_map.shape[0] and 0 <= ny < self.grid_map.shape[1]:
                    if self.grid_map[nx, ny] == FREESPACE_VALUE:  # 0 means free space
                        neighbors.append((nx, ny))
            return neighbors

        def reconstruct_path(came_from, current):
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        start = (int(round(start_point[0])), int(round(start_point[1])))
        end = (int(round(end_point[0])), int(round(end_point[1])))

        open_list = [start]
        came_from = {}

        g_score = np.full(self.grid_map.shape, np.inf)
        g_score[start] = 0

        f_score = np.full(self.grid_map.shape, np.inf)
        f_score[start] = heuristic(start, end)

        while open_list:
            # Select node with lowest f_score manually
            current = min(open_list, key=lambda pos: f_score[pos])

            if current == end:
                return reconstruct_path(came_from, current)

            open_list.remove(current)

            for neighbor in get_neighbors(current):
                dx = abs(neighbor[0] - current[0])
                dy = abs(neighbor[1] - current[1])
                if dx == 1 and dy == 1:
                    move_cost = 1.4  # diagonal
                else:
                    move_cost = 1    # straight

                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, end)
                    if neighbor not in open_list:
                        open_list.append(neighbor)

        return []  # no path found

    def path_following_pipeline(self, path):

        for target in path:
            while self.step() != -1:
                if self.follow_local_target(target):
                    break
        self.stop_motor()

    def get_pointcloud_2d(self):
        points = self.lidar.getPointCloud()
        points = np.array([[point.x, point.y] for point in points])
        points = points[~np.isinf(points).any(axis=1)]
        return points
    
    def transform_points_to_world(self, points_local):
        """
        Transform a batch of 2D points from robot-local frame to world frame using NumPy.

        Parameters:
            points_local: np.ndarray of shape (N, 2) — local [x, y] points

        Returns:
            points_world: np.ndarray of shape (N, 2) — transformed points in world coordinates
        """
        x_robot, y_robot = self.get_position()
        theta = self.get_heading('rad')  # yaw

        # Rotation matrix R(theta)
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        # Apply rotation and translation
        points_rotated = points_local @ R.T  # shape (N, 2)
        points_world = points_rotated + np.array([x_robot, y_robot])  # broadcast translation

        return points_world

    def get_pointcloud_world_coordinates(self):
        points_local = self.get_pointcloud_2d()
        points_world = self.transform_points_to_world(points_local)
        return points_world
    
    def convert_to_map_coordinate_matrix(self, points_world):
        # Compute transformation from world to map:
        # - Scaling (1 / RESOLUTION)
        # - Translation to shift origin to center of map

        # Rotation matrix (identity — no rotation needed in this case)
        R_map = np.array([
            [1 / RESOLUTION, 0],
            [0, -1 / RESOLUTION]  # Flip y-axis
        ])

        # Translation: move origin to center of map
        t_map = np.array([MAP_SIZE // 2, MAP_SIZE // 2])

        # Apply matrix transformation
        points_scaled = points_world @ R_map.T
        points_map = points_scaled + t_map

        return points_map.astype(np.int32)

    def draw_bresenham_line(self, map_target):
        x1, y1 = self.get_map_position()
        x2, y2 = map_target
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            self.grid_map[y1][x1] = FREESPACE_VALUE # Mark the line on the grid map
            if x1 == x2 and y1 == y2:
                break
            err2 = err * 2
            if err2 > -dy:
                err -= dy
                x1 += sx
            if err2 < dx:
                err += dx
                y1 += sy
                
    def bresenham_line(self, start, end):
        """
        Bresenham's line algorithm to generate points between two coordinates.
        Returns a list of (x, y) tuples.
        """
        x1, y1 = start
        x2, y2 = end
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            points.append((x1, y1))
            if x1 == x2 and y1 == y2:
                break
            err2 = err * 2
            if err2 > -dy:
                err -= dy
                x1 += sx
            if err2 < dx:
                err += dx
                y1 += sy

        return points
    
    def bresenham_to_obstacle_score(self, lidar_map_points):
        """
        Update log-odds map using Bresenham for each LIDAR point in map coordinates.
        """
        map_position = self.get_map_position()

        for map_target in lidar_map_points:
            points = self.bresenham_line(map_position, map_target)

            # Free points: all except the last
            for x, y in points[:-1]:
                if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
                    self.obstacle_score_map[y, x] -= 0.4
                    # self.grid_map[y, x] = FREESPACE_VALUE 

            # Occupied cell: last one
            x, y = points[-1]
            if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
                self.obstacle_score_map[y, x] += 0.85

    def update_grid_map(self):
        #  clip the score map from (-5, 5) to avoid overflow when applying exponential function np.exp
        limited_score_map = np.clip(self.obstacle_score_map, -5, 5) 
        P = 1 / (1 + np.exp(-limited_score_map))
        self.grid_map[P > 0.7] = OBSTACLE_VALUE
        self.grid_map[P < 0.5] = FREESPACE_VALUE

    def inflate_obstacles(self, grid_map, inflation_pixels=10):
        # Create a circular kernel based on the desired inflation radius
        kernel_size = int(2 * inflation_pixels + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Convert the grid map to uint8 format (0 and 255) for OpenCV processing
        grid_uint8 = (grid_map * 255).astype(np.uint8)
        cv2.imwrite('../../grid_map.png', grid_uint8)
        print(f"Unique values before dilation: {np.unique(grid_uint8)}")
        # Apply morphological dilation to expand obstacle areas
        inflated = cv2.dilate(grid_uint8, kernel, iterations=1)
        v, c = np.unique(inflated, return_counts=True)
        print(f"Unique values after dilation: {v}, Counts: {c}")

        # color_map = cv2.cvtColor(inflated, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
        cv2.imwrite('../../inflated_map_color.png', inflated)  # Save the colored image as PNG

        # Convert the result back to binary format (0 and 1)
        inflated_map = (inflated > 0).astype(np.uint8)
        
        return inflated_map

    def visualize_path_cv2(self, path, canvas_size=(1000, 1000), window_name="A* Path"):
        # Create a white canvas
        canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255

        # Draw the path in red
        for i in range(1, len(path)):
            pt1 = tuple(path[i - 1])
            pt2 = tuple(path[i])
            cv2.line(canvas, pt1, pt2, (0, 0, 255), thickness=2)

        # Mark start (green) and end (blue)
        if path:
            cv2.circle(canvas, tuple(path[0]), 5, (0, 255, 0), -1)   # Start: green
            cv2.circle(canvas, tuple(path[-1]), 5, (255, 0, 0), -1)  # End: blue

        # Show the canvas
        cv2.imshow(window_name, canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()