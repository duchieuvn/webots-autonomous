import numpy as np
from setup import setup_robot
import cv2
import random
from visualization import Visualizer
import pygame

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

    # def adapt_direction(self):
    #     def turn_right_milisecond(s=200):  
    #         self.set_robot_velocity(4, -4)
    #         self.step(s)

    #     def turn_left_milisecond(s=200):
    #         self.set_robot_velocity(-4, 4)
    #         self.step(s)

    #     def go_backward_milisecond(s=200):
    #         self.set_robot_velocity(-4, -4)
    #         self.step(s)

    #     # while there is an obstacle in front
    #     count = 0
    #     self.stop_motor()
    #     last_turn = 'right'
    #     distances = self.get_distances()
    #     while (min(distances[0], distances[2]) < 0.3 and count < 4):
    #         second = random.randint(100, 300)
    #         if distances[0] < distances[2]:
    #             turn_right_milisecond(second)
    #             last_turn = 'right'
    #         else:
    #             turn_left_milisecond(second)
    #             last_turn = 'left'

    #         count += 1

    #         distances = self.get_distances()
        
    #     if count == 4:
    #         # if there is an obstacle behind
    #         if (min(distances[1], distances[3]) > 0.3):
    #             go_backward_milisecond(200)
            
    #         if last_turn == 0:
    #             turn_left_milisecond(800)
    #         else:
    #             turn_right_milisecond(800)

    # with distance sensor and lidar ->

    def adapt_direction(self):
        def go_backward_millisecond(s=200):
            self.set_robot_velocity(-4, -4)
            self.step(s)

        # Stop first
        self.stop_motor()
        distances = self.get_distances()

        # Check if front is blocked
        front_blocked = min(distances[0], distances[2]) < 0.2
        back_clear = min(distances[1], distances[3]) > 0.2

        count = 0
        while front_blocked and count < 2:
            print(f"[DEBUG] Obstacle detected ahead. Attempting escape {count+1}/4")
            self.turn_toward_lidar_free_space(duration=random.randint(150, 400))
            self.stop_motor()
            self.step(100)  # pause
            distances = self.get_distances()
            front_blocked = min(distances[0], distances[2]) < 0.2
            count += 1

        # If still stuck, try to back up and turn again
        if count == 2 and back_clear:
            print("[DEBUG] Still stuck after 2 turns. Backing up.")
            go_backward_millisecond(700)
            self.turn_toward_lidar_free_space(duration=400)

    # LiDAR-only escape behavior: uses LiDAR to detect frontal obstacles ->

    def adapt_direction(self):
        """
        LiDAR-only escape behavior: uses LiDAR to detect frontal obstacles
        and turns toward free space. If still stuck after 2 turns, backs up.
        """
        def go_backward_millisecond(s=300):
            self.set_robot_velocity(-4, -4)
            self.step(s)

        self.stop_motor()
        front_blocked = self.lidar_front_blocked(threshold=0.25)

        count = 0
        while front_blocked and count < 3:
            print(f"[DEBUG] [LIDAR-ONLY] Obstacle ahead. Escape attempt {count+1}/2")
            self.turn_toward_lidar_free_space(duration=random.randint(200, 400))
            self.stop_motor()
            self.step(100)
            front_blocked = self.lidar_front_blocked(threshold=0.4)  # re-evaluate after turning
            count += 1

        if count == 3:
            print("[DEBUG] [LIDAR-ONLY] Still blocked after 2 turns. Backing up.")
            go_backward_millisecond(600)
            self.turn_toward_lidar_free_space(duration=400)


    def dwa_planner(self, world_target):
        MAX_SPEED = MAX_VELOCITY * self.wheel_radius
        v_samples = [0.08, 0.1, 0.15, 0.2, 0.5]
        w_samples = [0, 2, -2, 4, -4, 4.5, -4.5]

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
                    if predicted_distance > current_distance + 0.1:
                        good_path = False
                        break

                if not good_path:
                    continue

                predicted_angle_to_target = np.arctan2(world_target[1]-cy, world_target[0]-cx)
                heading_error = get_angle_diff(predicted_angle_to_target, ct)

                heading_score = np.cos(heading_error)
                distance_score = 1 - (predicted_distance / 2)
                speed_score = v / MAX_SPEED
                score = 4.0 * heading_score + 2.0 * distance_score + speed_score

                if score > best_score:
                    best_score = score
                    best_v = v
                    best_w = w

        return best_v, best_w

    def follow_local_target(self, map_target):
        if self.get_map_distance(map_target) < 2:
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
        while self.step(self.time_step) != -1 and count < 6000:
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

    def find_path(self, start_point, end_point):

        def generate_random_curved_path(start_point, end_point):
            step_size = 50
            deviation = 5
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
                offset_x = random.randint(-deviation, deviation)
                offset_y = random.randint(-deviation, deviation)
                x += offset_x
                y += offset_y
                points.append((int(round(x)), int(round(y))))

            filtered_points = []
            seen = set()
            for p in points:
                if p not in seen:
                    filtered_points.append(p)
                    seen.add(p)

            if filtered_points[-1] != (int(round(x1)), int(round(y1))):
                filtered_points[-1] = (int(round(x1)), int(round(y1)))

            return filtered_points

        def generate_path(start_point):
            path = []
            end_x, end_y = 400, 400

            for i in range(2):
                random_int = random.randint(200, 350)
                if i % 2 == 0:
                    end_x = start_point[0] + random_int
                else:
                    end_y = start_point[1] + random_int

                path += generate_random_curved_path(start_point, (end_x, end_y))
                start_point = [end_x + 30, end_y + 30]

            path += generate_random_curved_path(start_point, end_point)
            return path

        path = generate_path(start_point)
        return path

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
    
    def turn_toward_lidar_free_space(self, duration=300):
        """
        Uses LiDAR to choose the freer direction and turns toward it.
        """
        points = self.get_pointcloud_2d()

        if len(points) == 0:
            # Default to turning right if no LiDAR data
            self.set_robot_velocity(4, -4)
            self.step(duration)
            return

        left_points = points[points[:, 1] > 0]  # y > 0 → left
        right_points = points[points[:, 1] < 0]  # y < 0 → right

        threshold = 1.5
        left_clear = np.sum(np.linalg.norm(left_points, axis=1) > threshold)
        right_clear = np.sum(np.linalg.norm(right_points, axis=1) > threshold)

        if left_clear > right_clear:
            print("[DEBUG] LiDAR turning LEFT")
            self.set_robot_velocity(-4, 4)  # Turn left
        else:
            print("[DEBUG] LiDAR turning RIGHT")
            self.set_robot_velocity(4, -4)  # Turn right

        self.step(duration)

    def lidar_front_blocked(self, threshold=0.4):
        points = self.get_pointcloud_2d()
        if len(points) == 0:
            return False

        # Filter front-facing points: -30° to +30° (x > 0.1 and |y| < 0.3)
        front_points = points[(points[:, 0] > 0.1) & (np.abs(points[:, 1]) < 0.06)]
        return np.any(np.linalg.norm(front_points, axis=1) < threshold)
