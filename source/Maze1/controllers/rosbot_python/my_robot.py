import numpy as np
from setup import setup_robot
import cv2
import random
from visualization import Visualizer
import pygame
import time
from sklearn.cluster import DBSCAN

TIME_STEP = 32
MAX_VELOCITY = 26
WHEEL_RADIUS = 0.043
AXLE_LENGTH = 0.18
# MAP_SIZE = 500 # 10m x 10.0m grid map
RESOLUTION = 0.03 # 5cm resolution
MAP_SIZE = int(10.0 / RESOLUTION)
# 1000 pixels = 10m, so each pixel is 0.01m = 1cm


OBSTACLE_VALUE = 1
FREESPACE_VALUE = 0
UNKNOWN_VALUE = 255

grid_map = np.full((MAP_SIZE, MAP_SIZE), UNKNOWN_VALUE, dtype=np.uint8)
# grid_map = np.full((MAP_SIZE, MAP_SIZE), FREESPACE_VALUE, dtype=np.uint8)
obstacle_score_map = np.full((MAP_SIZE, MAP_SIZE), 0.0, dtype=np.float32)
# obstacle_score_map = np.full((MAP_SIZE, MAP_SIZE), 2, dtype=np.float32)

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
        # self.frontiers = []

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
                    # if self.there_is_obstacle([predicted_map_x, predicted_map_y]):
                    if not self.is_safe_cell((predicted_map_x, predicted_map_y)):
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
        current_map_pos = self.get_map_position()

        # Consider target reached if within 6 pixels (~6 cm)
        if np.linalg.norm(np.array(current_map_pos) - np.array(map_target)) < 6:
            self.stop_motor()
            return True

        # Frontier got blocked (e.g. due to poor lidar or mapping noise)
        if self.grid_map[map_target[1], map_target[0]] == OBSTACLE_VALUE:
            return True

        # If too close to wall (unsafe to proceed), skip it
        if not self.is_safe_cell(map_target, inflation_pixels=6, fallback_lidar=True):
            # print(f"Skipping frontier {map_target}, not safe.")
            return True

        # Otherwise, compute motion toward target
        world_target = self.convert_to_world_coordinates(*map_target)
        v, w = self.dwa_planner(world_target)
        left_speed, right_speed = self.velocity_to_wheel_speeds(v, w)
        self.set_robot_velocity(left_speed, right_speed)
        return False



    def there_is_obstacle(self, map_target):
        if self.grid_map[map_target[1], map_target[0]] == OBSTACLE_VALUE:
            return True
        return False
    
    def is_safe_cell(self, map_target, inflation_pixels=6, fallback_lidar=True):
        """
        Checks whether a map cell is safe for navigation, using an inflated obstacle map.
        This prevents the robot from moving too close to walls or corners.

        Args:
            map_target: (x, y) tuple in map coordinates
            inflation_pixels: how much margin around obstacles to consider unsafe

        Returns:
            True if the cell is free and outside inflated danger zone, False otherwise
        """
        x, y = map_target
        if not (0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE):
            return False

        # Check inflated obstacle map
        if not hasattr(self, '_inflated_map_cache') or self._inflated_map_cache_inflation != inflation_pixels:
            self._inflated_map_cache = self.inflate_obstacles(self.grid_map, inflation_pixels)
            self._inflated_map_cache_inflation = inflation_pixels

        is_safe = self._inflated_map_cache[y, x] == 0
        if is_safe:
            return True  

        # If fallback allowed, check with LiDAR clearance radius
        if fallback_lidar:
            return self.is_frontier_clear_by_lidar((x, y), min_clearance_cm=10)

        return False


    
    def explore(self):
        vis = Visualizer(map_size=MAP_SIZE)
        
        running = True
        count = 0
        frontiers = []
        last_frontier_update = -100
        self.current_frontier_target = None
        frontier_timer = 0
        frontier_start_time = None
        FRONTIER_TIMEOUT = 5.0  # seconds

        while self.step(self.time_step) != -1 and count < 1500:
            for event in pygame.event.get(): 
                if event.type == pygame.QUIT:
                    running = False
            vis.clear_screen()

            distances = self.get_distances()
            if self.is_front_blocked_lidar():
                self.adapt_direction()
            else:
                self.set_robot_velocity(8, 8)
            # if min(distances[0], distances[2]) < 0.3:
            #     self.adapt_direction()
            # else:
            #     self.set_robot_velocity(8, 8)
            
            if count % 30 == 0:
                unique = np.unique(self.grid_map, return_counts=True)
                # print(f"Grid values: {dict(zip(unique[0], unique[1]))}")

            if count % 30 == 0:
                map_x, map_y = self.get_map_position()
                patch = self.grid_map[map_y-5:map_y+6, map_x-5:map_x+6]
                # print("Region around robot:\n", patch)
            if self.current_frontier_target is None:
                points = self.get_pointcloud_world_coordinates()
                map_points = self.convert_to_map_coordinate_matrix(points)
                
                if count % 20 == 0 and not self.is_turning():
                    # for map_point in map_points:
                        # self.draw_bresenham_line(map_point)
                        self.bresenham_to_obstacle_score(map_points)
                        self.update_grid_map()
                        # print(self.obstacle_score_map[100:200, 500:700])
                        # vis.draw_line(cur_map_pos, map_point)
            
                if self.current_frontier_target is None and count - last_frontier_update > 30:
                    frontiers = self.detect_frontiers()

                    # Clamp to valid map bounds
                    frontiers = [
                        f for f in frontiers
                        if 0 <= f[0] < MAP_SIZE and 0 <= f[1] < MAP_SIZE
                    ]

                    robot_pos = self.get_map_position()

                    # Filter: free space only & not too close
                    frontiers = [
                        f for f in frontiers
                        if np.linalg.norm(np.array(f) - robot_pos) > 50 and self.grid_map[f[1], f[0]] == FREESPACE_VALUE
                    ]

                    # Sort by distance and limit to first 200
                    frontiers.sort(key=lambda p: np.linalg.norm(np.array(p) - robot_pos))
                    frontiers = frontiers[:200]
                    # Filter valid frontiers
                    frontiers = [f for f in frontiers if self.grid_map[f[1], f[0]] == FREESPACE_VALUE and self.is_frontier_clear_by_lidar(f, min_clearance_cm=5)]
                    
                    if frontiers:
                        scored = sorted(frontiers, key=lambda f: self.score_frontier(f, robot_pos))
                        self.current_frontier_target = scored[0]
                        frontier_start_time = time.time()
                        last_frontier_update = count
                    else:
                        self.current_frontier_target = None
                #     print(f"Number of frontiers: {len(frontiers)}")
                # print(f"Target frontier: {self.current_frontier_target}, Robot: {self.get_map_position()}")
            
            # Follow the current frontier target
            if self.current_frontier_target is not None and not self.is_turning():
                x, y = self.current_frontier_target
                if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
                    done = self.follow_local_target(self.current_frontier_target)
                    frontier_timer += 1
                    # Frontier completion or timeout or newly blocked
                    if (done 
                        or frontier_timer > 150 
                        or self.grid_map[y, x] == OBSTACLE_VALUE 
                        or (frontier_start_time is not None and time.time() - frontier_start_time > FRONTIER_TIMEOUT)):
                        self.current_frontier_target = None
                        frontier_timer = 0
                else:
                    self.current_frontier_target = None
                    frontier_timer = 0    

            

            vis.update_screen_with_map(self.grid_map)
            vis.draw_robot(self.get_map_position())
            # vis.draw_frontiers(frontiers[:50])
            # filtered_frontiers = [f for f in frontiers if self.is_frontier_clear_by_lidar(f)]
            # vis.draw_filtered_frontiers(filtered_frontiers)
            if self.current_frontier_target is not None:
                vis.draw_centroids([self.current_frontier_target])
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
    
    # def bresenham_to_obstacle_score(self, lidar_map_points):
    #     """
    #     Update log-odds map using Bresenham for each LIDAR point in map coordinates.
    #     """
    #     map_position = self.get_map_position()

    #     for map_target in lidar_map_points:
    #         points = self.bresenham_line(map_position, map_target)

    #         # Free points: all except the last
    #         for x, y in points[:-1]:
    #             if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
    #                 self.obstacle_score_map[y, x] -= 0.4
    #                 # self.grid_map[y, x] = FREESPACE_VALUE 

    #         # Occupied cell: last one
    #         x, y = points[-1]
    #         if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
    #             self.obstacle_score_map[y, x] += 0.85
    
    def bresenham_to_obstacle_score(self, lidar_map_points):
        """
        Update log-odds map using Bresenham for each LIDAR point in map coordinates.
        Only update unknown (255) areas to preserve frontier boundaries.
        """
        map_position = self.get_map_position()

        for map_target in lidar_map_points:
            points = self.bresenham_line(map_position, map_target)

            max_range = 100  # ~100 pixels = 1m
            count = 0

            for x, y in points[:-1]:
                if count > max_range:
                    break  # stop updating too deep into unknowns
                if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
                    if self.grid_map[y, x] == UNKNOWN_VALUE:
                        self.obstacle_score_map[y, x] -= 0.4
                count += 1

            # Final point = obstacle
            x, y = points[-1]
            if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
                if self.grid_map[y, x] == UNKNOWN_VALUE:
                    self.obstacle_score_map[y, x] += 0.85

    def update_grid_map(self):
        #  clip the score map from (-5, 5) to avoid overflow when applying exponential function np.exp
        limited_score_map = np.clip(self.obstacle_score_map, -5, 5) 
        P = 1 / (1 + np.exp(-limited_score_map))
        self.grid_map[P > 0.65] = OBSTACLE_VALUE
        self.grid_map[P < 0.35] = FREESPACE_VALUE

    def inflate_obstacles(self, grid_map, inflation_pixels=10):
        # Create a circular kernel based on the desired inflation radius
        kernel_size = int(2 * inflation_pixels + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Convert the grid map to uint8 format (0 and 255) for OpenCV processing
        grid_uint8 = (grid_map * 255).astype(np.uint8)
        cv2.imwrite('../../grid_map.png', grid_uint8)
        # print(f"Unique values before dilation: {np.unique(grid_uint8)}")
        # Apply morphological dilation to expand obstacle areas
        inflated = cv2.dilate(grid_uint8, kernel, iterations=1)
        v, c = np.unique(inflated, return_counts=True)
        # print(f"Unique values after dilation: {v}, Counts: {c}")

        # color_map = cv2.cvtColor(inflated, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
        cv2.imwrite('../../inflated_map_color.png', inflated)  # Save the colored image as PNG

        # Convert the result back to binary format (0 and 1)
        inflated_map = (inflated > 0).astype(np.uint8)
        
        return inflated_map
    

    def detect_frontiers(self):
        frontiers = []
        for y in range(1, MAP_SIZE - 1):
            for x in range(1, MAP_SIZE - 1):
                if self.grid_map[y, x] == FREESPACE_VALUE:
                    neighbors = self.grid_map[y-1:y+2, x-1:x+2]
                    if UNKNOWN_VALUE in neighbors:
                        frontiers.append((x, y))
        return frontiers

    def cluster_frontiers(self, frontiers, eps=5, min_samples=3):
        if not frontiers:
            return []

        frontiers_np = np.array(frontiers)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(frontiers_np)
        labels = clustering.labels_

        # Group frontier points by cluster label
        clusters = {}
        for point, label in zip(frontiers, labels):
            if label == -1:
                continue  # skip noise
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(point)

        # Compute centroids of each cluster
        centroids = []
        for cluster_points in clusters.values():
            xs, ys = zip(*cluster_points)
            centroid = (int(np.mean(xs)), int(np.mean(ys)))
            centroids.append(centroid)

        return centroids
    
    def is_valid_frontier(self, f, min_clearance=3):
        x, y = f
        if not (0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE):
            return False

        # Check a slightly larger patch for safety margin (adjustable)
        patch = self.grid_map[max(0, y - min_clearance):y + min_clearance + 1,
                            max(0, x - min_clearance):x + min_clearance + 1]
        
        # Ensure at least some unknown space around but not fully surrounded by obstacles
        num_obs = np.count_nonzero(patch == OBSTACLE_VALUE)
        num_total = patch.size
        if num_obs > 0.3 * num_total:
            return False

        # Optionally keep front-facing only (can comment this out for early testing)
        robot_x, robot_y = self.get_map_position()
        heading = self.get_heading('rad')
        dx = x - robot_x
        dy = robot_y - y  # y axis flipped
        angle_to_point = np.arctan2(dy, dx)
        angle_diff = get_angle_diff(angle_to_point, heading)

        return abs(angle_diff) < np.pi * 0.75  # Allow wider cone
    
    def is_frontier_clear_by_lidar(self, map_point, min_clearance_cm=20):
        """
        Check if a frontier is clear of obstacles within a circular radius using LiDAR data.
        """
        world_x, world_y = self.convert_to_world_coordinates(*map_point)
        robot_x, robot_y = self.get_position()

        # Get all LiDAR points in world frame
        points_world = self.get_pointcloud_world_coordinates()

        for px, py in points_world:
            dist = np.hypot(px - world_x, py - world_y)
            if dist < min_clearance_cm / 100.0:  # convert cm to meters
                return False
        return True
    
    def is_front_blocked_lidar(self, min_clearance=0.25):
        points = self.get_pointcloud_2d()
        forward_points = points[(points[:, 0] > 0) & (np.abs(points[:, 1]) < 0.2)]
        return np.any(np.hypot(forward_points[:, 0], forward_points[:, 1]) < min_clearance)
    
    def score_frontier(self, f, robot_pos):
        dist = np.linalg.norm(np.array(f) - robot_pos)
        heading = self.get_heading('rad')
        dx, dy = f[0] - robot_pos[0], robot_pos[1] - f[1]
        angle_to = np.arctan2(dy, dx)
        angle_diff = abs(get_angle_diff(angle_to, heading))
        angle_score = np.cos(angle_diff)
        
        wall_proximity_penalty = 0
        if not self.is_safe_cell(f, inflation_pixels=6, fallback_lidar=True):
            wall_proximity_penalty = 9999
        
        return dist - 40 * angle_score + wall_proximity_penalty  # Tune weights
    
    def astar_path(self, start, goal):
        from queue import PriorityQueue

        def heuristic(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))

        open_set = PriorityQueue()
        open_set.put((0, start))
        came_from = {}
        g_score = {start: 0}

        while not open_set.empty():
            _, current = open_set.get()
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]:
                neighbor = (current[0]+dx, current[1]+dy)
                if not (0 <= neighbor[0] < MAP_SIZE and 0 <= neighbor[1] < MAP_SIZE):
                    continue
                # if self.grid_map[neighbor[1], neighbor[0]] == OBSTACLE_VALUE:
                #     continue
                inflated = self.inflate_obstacles(self.grid_map, inflation_pixels=5)
                if inflated[neighbor[1], neighbor[0]] != 0:
                    continue  # Cell is too close to an obstacle

                tentative = g_score[current] + heuristic(current, neighbor)
                if neighbor not in g_score or tentative < g_score[neighbor]:
                    g_score[neighbor] = tentative
                    f_score = tentative + heuristic(neighbor, goal)
                    open_set.put((f_score, neighbor))
                    came_from[neighbor] = current

        return []  # No path found
    
    def follow_path(self, path):
        if not path:
            return True

        for waypoint in path:
            while self.step(self.time_step) != -1:
                if self.is_front_blocked_lidar():
                    self.adapt_direction()
                    break
                if not self.is_safe_cell(waypoint, inflation_pixels=5):
                    return True  # Path blocked, try another frontier
                done = self.follow_local_target(waypoint)
                if done:
                    break
        return True
    
    
