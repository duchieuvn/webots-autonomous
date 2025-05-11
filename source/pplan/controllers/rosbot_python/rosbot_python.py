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

OBSTACLE_VALUE = 255

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
        dir_y = orientation[1]
        angle_rad = np.arctan2(dir_y, dir_x)
        if type == 'rad':
            return angle_rad
        elif type == 'deg':
            angle_deg = (np.degrees(angle_rad) + 360) % 360
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
        r = 10

        intersect_list = []
        for x_border in [-r, r]:
            for j in range(-r, r+1):
                point_x = map_x + x_border
                point_y = map_y + j
                if self.grid_map[point_y, point_x] == 100 and self.get_angle_diff([point_x, point_y]) <= 90:
                    visual_map[point_y, point_x] = (0,0,255)
                    intersect_list.append(np.array([point_x, point_y]))

        for y_border in [-r, r]:
            for i in range(-r, r+1):
                point_x = map_x + i
                point_y = map_y + y_border
                if self.grid_map[point_y, point_x] == 100 and self.get_angle_diff([point_x, point_y]) <= 90:
                    visual_map[point_y, point_x] = (0,0,255)
                    intersect_list.append(np.array([point_x, point_y]))

        if len(intersect_list) == 0:
            print("No intersection found")
            return None
        else:
            dist = [np.linalg.norm(intersect_point - np.array([map_x, map_y])) for intersect_point in intersect_list]
            return intersect_list[np.argmin(dist)]

    def there_is_obstacle(self, map_target):
        if self.grid_map[map_target[1], map_target[0]] == OBSTACLE_VALUE:
            return True
        return False

    def dwa_planner(self, world_target):
        MAX_SPEED = MAX_VELOCITY * self.wheel_radius
        
        v_samples = [0.08, 0.1, 0.15, 0.2, 0.5]
        # w_samples = [0.2, 0.5, 1, 1.5]
        w_samples = [0.2, -0.2, 0.5, -0.5, 0, 1, -1, 1.5, -1.5]
        # w_samples += [-w for w in w_samples]

        best_score = -float('inf')
        min_error = float('inf')
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

                error = 4*np.sin(heading_error) + 2*predicted_distance + 0.5*(MAX_SPEED - v)
                if error < min_error:

                    print(f"v: {v}, w: {w}")
                    print(f"Predicted angle to target: {np.degrees(predicted_angle_to_target)}")
                    print(f"Heading error: {np.degrees(heading_error)}")
                    print(f"Distance: {current_distance} {predicted_distance}")
                    print(f"New min error: {error}")
                    min_error = error
                    best_v = v
                    best_w = w



        return best_v, best_w

    def dwa_plan(self, map_target, tolerance=0.2):
        # distance = np.linalg.norm(map_target - current_pos)

        # if distance < tolerance:
        #     print("[dwa_plan] Reached target!")
        #     self.stop_motor()
        #     return

        world_target = self.convert_to_world_coordinates(map_target[0], map_target[1])
        v, w = self.dwa_planner(world_target)

        left_speed, right_speed = self.velocity_to_wheel_speeds(v, w)
        print("Left speed: ", left_speed, "Right speed: ", right_speed)

        self.set_robot_velocity(left_speed, right_speed)

def get_angle_diff(a, b):
        diff = a - b
        while diff > np.pi:
            diff -= 2*np.pi
        while diff < -np.pi:
            diff += 2*np.pi
        return diff


robot = MyRobot()

def main():
    try:
        pcount = 0
        last_target = robot.get_map_position()
        while robot.step(TIME_STEP) != -1:
            _target = robot.get_local_target()
            target = _target if _target is not None else last_target
            last_target = target
            robot.draw_position_in_map(visual_map)

            map_resized = cv2.resize(visual_map, (500, 500))
            cv2.imshow("Occupancy Grid Map", map_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            print("Target: ", target)
            robot.dwa_plan(target)
            print('---------------------------')
            # time.sleep(1)

    except Exception as e:
        print("An error occurred:", e)
        cv2.destroyAllWindows()
        
main()