from controller import Robot, Keyboard, Supervisor
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2

TIME_STEP = 32
MAX_VELOCITY = 26

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
grid_map = np.array([[0 for _ in range(MAP_SIZE)] for _ in range(MAP_SIZE)], dtype=np.uint8)

def turn_right_milisecond(s=200):
    front_left_motor.setVelocity(4)
    rear_left_motor.setVelocity(4)
    front_right_motor.setVelocity(-4)
    rear_right_motor.setVelocity(-4) 
    robot.step(s)  

def turn_left_milisecond(s=200):
    front_left_motor.setVelocity(-4)
    rear_left_motor.setVelocity(-4)
    front_right_motor.setVelocity(4)
    rear_right_motor.setVelocity(4) 
    robot.step(s)  

def go_backward_milisecond(s=200):
    front_left_motor.setVelocity(-4)
    rear_left_motor.setVelocity(-4)
    front_right_motor.setVelocity(-4)
    rear_right_motor.setVelocity(-4) 
    robot.step(s)

def stop_motor():
    front_left_motor.setVelocity(0)
    rear_left_motor.setVelocity(0)
    front_right_motor.setVelocity(0)
    rear_right_motor.setVelocity(0) 

def adapt_direction(distance_sensors_value):
    # while there is an obstacle in front
    count = 0
    stop_motor()
    last_turn = 'right'
    while (
    min(distance_sensors_value[0], distance_sensors_value[2]) < 0.3 
        and count < 4
    ):
        second = random.randint(100, 300)
        if distance_sensors_value[0] < distance_sensors_value[2]:
            turn_right_milisecond(second)
            last_turn = 'right'
        else:
            turn_left_milisecond(second)
            last_turn = 'left'

        count += 1

        for i in range(4):
            distance_sensors_value[i] = distance_sensors[i].getValue()
    
    if count == 4:
        # if there is an obstacle behind
        if (min(distance_sensors_value[1], distance_sensors_value[3]) > 0.3):
            go_backward_milisecond(200)
        
        if last_turn == 0:
            turn_left_milisecond(800)
        else:
            turn_right_milisecond(800)

def calculate_velocity(distance_sensors_value):
    coefficients = [[15.0, -9.0], [-15.0, 9.0]]
    motor_speed = [0.0, 0.0]
    for i in range(2):
        avoidance_speed = 0.0
        for j in range(1, 3):
            avoidance_speed += (2.0 - distance_sensors_value[j]) ** 2 * coefficients[i][j - 1]
        motor_speed[i] = 6.0 + avoidance_speed
        motor_speed[i] = min(motor_speed[i], MAX_VELOCITY)
    return motor_speed

def get_distances():
    distance_sensors_value = []
    for i in range(4):
            distance_sensors_value.append(distance_sensors[i].getValue())
      
    return distance_sensors_value

def draw_position_in_map():
    pos = np.array(robot.getSelf().getPosition()[:2]) / RESOLUTION
    map_x = int(MAP_SIZE//2 + int(pos[0]))
    map_y = int(MAP_SIZE//2 - np.ceil(pos[1]))
    for i in [-3, -2, -1, 0, 1, 2, 3]:
        for j in [-3, -2, -1, 0, 1, 2, 3]:
            grid_map[map_y+j][map_x+i] = 255

# plt.ion()
# fig, ax = plt.subplots()
g_count = 0 

# Main loop
try:
    while robot.step(TIME_STEP) != -1:

        pos = np.array(robot.getSelf().getPosition()[:2]) / RESOLUTION
        map_x = int(MAP_SIZE//2 + int(pos[0]))
        map_y = int(MAP_SIZE//2 - np.ceil(pos[1]))
        for i in [-3, -2, -1, 0, 1, 2, 3]:
            for j in [-3, -2, -1, 0, 1, 2, 3]:
                grid_map[map_y+j][map_x+i] = 255

        distance_sensors_value = get_distances()
        adapt_direction(distance_sensors_value)

        motor_speed = [8, 8]

        front_left_motor.setVelocity(motor_speed[0])
        front_right_motor.setVelocity(motor_speed[1])
        rear_left_motor.setVelocity(motor_speed[0])
        rear_right_motor.setVelocity(motor_speed[1])

        # Hiển thị bản đồ
        if g_count % 100 == 0:
            np.savetxt("../array.csv", grid_map, delimiter=",", fmt="%d")
            map_img = Image.fromarray(grid_map, mode='L')
            map_img.save("../map.png")


        map_resized = cv2.resize(grid_map, (500, 500))  
        cv2.imshow("Occupancy Grid Map", map_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        g_count += 1

except:
    pass
