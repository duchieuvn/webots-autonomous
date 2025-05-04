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
WHEEL_RADIUS = 0.043  # meters
AXLE_LENGTH = 0.18    # meters (distance between left and right wheels)

MAP_SIZE = 1000 # 1000cm = 10m
RESOLUTION = 0.01 # 1cm for 1 grid cell
grid_map = cv2.imread('../../textures/path_test_map.bmp', cv2.IMREAD_GRAYSCALE)
visual_map = cv2.imread('../../textures/map1.png', cv2.IMREAD_COLOR)
visual_map = cv2.cvtColor(visual_map, cv2.COLOR_BGR2RGB)

class MyRobot:
    def __init__(self):
        self.robot, self.motors, self.wheel_sensors, self.imu, self.camera_rgb, self.camera_depth, self.lidar, self.distance_sensors = setup_robot()
        self.grid_map = grid_map

    def step(self, timestep):
        return self.robot.step(timestep)

    def stop_motor(self):
        for motor in self.motors.values():
            motor.setVelocity(0)

    def set_motor_velocity(self, fl, fr, rl, rr):
        self.motors['fl'].setVelocity(fl)
        self.motors['fr'].setVelocity(fr)
        self.motors['rl'].setVelocity(rl)
        self.motors['rr'].setVelocity(rr)

    def go_backward_millisecond(self, ms=200):
        self.set_motor_velocity(-4, -4, -4, -4)
        self.step(ms)

    def get_heading(self, type='deg'):
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

    def draw_position_in_map(self, visual_map):
        pos = self.get_position()
        map_x, map_y = self.convert_to_map_coordinates(pos[0], pos[1])
        for i in range(-3, 4):
            for j in range(-3, 4):
                visual_map[map_y + j][map_x + i] = 255

    def get_local_target(self, last_target=None):
        map_x, map_y = self.get_map_position()
        r = 10

        intersect_list = []
        for x_border in [-r, r]:
            for j in range(-r, r+1):
                point_x = map_x + x_border
                point_y = map_y + j
                if self.grid_map[point_y, point_x] == 100 and self.get_angle_diff([point_x, point_y]) <= 45:
                    visual_map[point_y, point_x] = (0,0,255)
                    intersect_list.append(np.array([point_x, point_y]))

        for y_border in [-r, r]:
            for i in range(-r, r+1):
                point_x = map_x + i
                point_y = map_y + y_border
                if self.grid_map[point_y, point_x] == 100 and self.get_angle_diff([point_x, point_y]) <= 45:
                    visual_map[point_y, point_x] = (0,0,255)
                    intersect_list.append(np.array([point_x, point_y]))

        if len(intersect_list) == 0:
            print("No intersection found")
            return last_target
        else:
            dist = [np.linalg.norm(intersect_point - np.array([map_x, map_y])) for intersect_point in intersect_list]
            print('--',intersect_list[np.argmin(dist)])
            return intersect_list[np.argmin(dist)]

    def dwa_planner(self, map_target, config):
        max_v = config['max_v']
        max_w = config['max_w']
        v_samples = config['v_samples']
        w_samples = config['w_samples']
        predict_time = config['predict_time']
        robot_radius = config['robot_radius']
        dt = config['dt']

        best_score = -float('inf')
        best_v = 0.0
        best_w = 0.0

        x, y = self.get_map_position()
        theta = self.get_heading('rad')

        for v in np.linspace(0.01, max_v, v_samples):
            for w in np.linspace(-max_w, max_w, w_samples):
                cx, cy, ct = x, y, theta
                collision = False
                for _ in np.arange(0, predict_time, dt):
                    cx += v * np.cos(ct) * dt
                    cy += v * np.sin(ct) * dt
                    ct += w * dt

                    distance_sensor_data = self.get_distances()
                    if min(distance_sensor_data[0], distance_sensor_data[1]) < robot_radius:
                        collision = True
                        break

                if collision:
                    continue

                heading_diff = self.get_angle_diff(map_target)
                score = -heading_diff + 0.5 * v

                if score > best_score:
                    best_score = score
                    best_v = v
                    best_w = w

        return best_v, best_w

    def dwa_plan(self, map_target, tolerance=0.2):
        current_pos = self.get_map_position()
        distance = np.linalg.norm(map_target - current_pos)

        if distance < tolerance:
            print("[dwa_plan] Reached target!")
            self.stop_motor()
            return

        config = {
            'max_v': 0.3,
            'max_w': 1.5,
            'v_samples': 3,
            'w_samples': 3,
            'predict_time': 1.0,
            'dt': 0.1,
            'robot_radius': 0.3
        }

        v, w = self.dwa_planner(map_target, config)

        left_speed, right_speed = velocity_to_wheel_speeds(v, w, AXLE_LENGTH, WHEEL_RADIUS)
        print("Left speed: ", left_speed, "Right speed: ", right_speed)

        self.set_motor_velocity(left_speed, right_speed, left_speed, right_speed)

robot = MyRobot()

def velocity_to_wheel_speeds(v, w, wheel_base, wheel_radius):
    v_left = v + (wheel_base / 2.0) * w
    v_right = v - (wheel_base / 2.0) * w
    left_speed = v_left / wheel_radius
    right_speed = v_right / wheel_radius
    return left_speed, right_speed

def main():
    try:
        pcount = 0
        last_target = robot.get_map_position()
        while robot.step(TIME_STEP) != -1:
            target = robot.get_local_target(last_target)
            last_target = target if target is not None else last_target
            print("Target: ", target)
            robot.draw_position_in_map(visual_map)

            map_resized = cv2.resize(visual_map, (500, 500))
            cv2.imshow("Occupancy Grid Map", map_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            robot.dwa_plan(target)
            # time.sleep(0.1)

    except Exception as e:
        print("An error occurred:", e)
        cv2.destroyAllWindows()
        
main()