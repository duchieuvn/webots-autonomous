from controller import Robot, Keyboard, Supervisor
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

# Initialize the robot
robot = Supervisor()

# Get motor devices
front_left_motor = robot.getDevice("fl_wheel_joint")
front_right_motor = robot.getDevice("fr_wheel_joint")
rear_left_motor = robot.getDevice("rl_wheel_joint")
rear_right_motor = robot.getDevice("rr_wheel_joint")
front_left_motor.setPosition(float('inf'))
front_right_motor.setPosition(float('inf'))
rear_left_motor.setPosition(float('inf'))
rear_right_motor.setPosition(float('inf'))
front_left_motor.setVelocity(0.0)
front_right_motor.setVelocity(0.0)
rear_left_motor.setVelocity(0.0)
rear_right_motor.setVelocity(0.0)

# Get position sensors and enable them
front_left_position_sensor = robot.getDevice("front left wheel motor sensor")
front_right_position_sensor = robot.getDevice("front right wheel motor sensor")
rear_left_position_sensor = robot.getDevice("rear left wheel motor sensor")
rear_right_position_sensor = robot.getDevice("rear right wheel motor sensor")
front_left_position_sensor.enable(TIME_STEP)
front_right_position_sensor.enable(TIME_STEP)
rear_left_position_sensor.enable(TIME_STEP)
rear_right_position_sensor.enable(TIME_STEP)

# Get RGB and depth cameras and enable them
camera_rgb = robot.getDevice("camera rgb")
camera_depth = robot.getDevice("camera depth")
camera_rgb.enable(TIME_STEP)
camera_depth.enable(TIME_STEP)

# Get Lidar device and enable it
lidar = robot.getDevice("laser")
lidar.enable(TIME_STEP)
lidar.enablePointCloud()

# Get IMU devices and enable them
accelerometer = robot.getDevice("imu accelerometer")
gyro = robot.getDevice("imu gyro")
compass = robot.getDevice("imu compass")
accelerometer.enable(TIME_STEP)
gyro.enable(TIME_STEP)
compass.enable(TIME_STEP)

# Get distance sensors and enable them
distance_sensors = []
distance_sensors.append(robot.getDevice("fl_range"))
distance_sensors.append(robot.getDevice("rl_range"))
distance_sensors.append(robot.getDevice("fr_range"))
distance_sensors.append(robot.getDevice("rr_range"))
for sensor in distance_sensors:
    sensor.enable(TIME_STEP)

MAP_SIZE = 1000 # 1000cm = 10m
RESOLUTION = 0.01 # 1cm for 1 grid cell
# grid_map = np.array([[0 for _ in range(MAP_SIZE)] for _ in range(MAP_SIZE)], dtype=np.uint8)
grid_map = cv2.imread('../../textures/path_test_map.bmp', cv2.IMREAD_GRAYSCALE)
visual_map = cv2.imread('../../textures/map1.png', cv2.IMREAD_COLOR)
visual_map = cv2.cvtColor(visual_map, cv2.COLOR_BGR2RGB)

try:
    pcount = 0  
    while robot.step(TIME_STEP) != -1 and pcount < 1:
        heading = robot.getSelf().getOrientation()
        print('heading', heading)

        pcount += 1


except Exception as e:
    print(f"An error occurred: {e}")
    exit(0)
    pass
