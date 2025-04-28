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

def get_heading_deg():
    """
    Convert compass vector to heading angle in degrees (0° = North, increasing clockwise).
    """
    compass_values = compass.getValues()
    rad = np.arctan2(compass_values[1], compass_values[0])  # y/x
    deg = np.degrees(rad)
    heading = (deg + 360) % 360
    return heading

def rotate_in_place(degree, direction, speed=5.0):
    
    # --- Get initial heading ---
    initial_heading = get_heading_deg()
    target_heading = (initial_heading + degree) % 360

    # --- Determine rotation direction ---
    sign = 1 if (direction == 'right') else -1
    front_left_motor.setVelocity(speed * sign)
    rear_left_motor.setVelocity(speed * sign)
    front_right_motor.setVelocity(-speed * sign)
    rear_right_motor.setVelocity(-speed * sign)

    def angle_diff(current, target):
        diff = (target - current + 540) % 360 - 180
        return abs(diff)

    # --- Rotate until reaching target heading ---
    while robot.step(TIME_STEP) != -1:
        current_heading = get_heading_deg()
        diff = angle_diff(current_heading, target_heading)
        if diff < 2.0:  
            break

    stop_motor()

    # Reference from ChatGPT: https://chatgpt.com/share/68011098-b5d4-8013-be07-b764d0f7f035

def adapt_direction(distance_sensors_value):
    # while there is an obstacle in front
    count = 0
    stop_motor()
    last_turn = 'right'
    while (
    min(distance_sensors_value[0], distance_sensors_value[2]) < 0.3 
        and count < 4
    ):
        degree = random.randint(30, 90)
        if distance_sensors_value[0] < distance_sensors_value[2]:
            rotate_in_place(degree, 'right')
            last_turn = 'right'
        else:
            rotate_in_place(degree, 'left')
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

def convert_to_map_coordinates(x, y):
    # Convert Webots coordinates to map coordinates
    map_x = MAP_SIZE//2 + int(x / RESOLUTION)
    map_y = MAP_SIZE//2 - np.ceil(y / RESOLUTION)
    return int(map_x), int(map_y)

def draw_position_in_map():
    pos = np.array(robot.getSelf().getPosition()[:2])
    map_x, map_y = convert_to_map_coordinates(pos[0], pos[1])
    for i in [-3, -2, -1, 0, 1, 2, 3]:
        for j in [-3, -2, -1, 0, 1, 2, 3]:
            visual_map[map_y+j][map_x+i] = 255

def get_position():
    pos = robot.getSelf().getPosition()
    return np.array([pos[0], pos[1]])

def get_map_position():
    pos = get_position()
    map_x, map_y = convert_to_map_coordinates(pos[0], pos[1])
    return np.array([map_x, map_y])

def simple_plan(map_target, tolerance=0.2):

    def angle_to_target(current_pos, target_pos):
        # Compute angle (in degrees) from current position to target position
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        angle = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
        return angle

    def rotate_to_heading(target_heading, threshold=3.0):
        # Rotate the robot to face the target heading (compass-based)
        while robot.step(TIME_STEP) != -1:
            current_heading = get_heading_deg()
            diff = (target_heading - current_heading + 540) % 360 - 180
            
            if abs(diff) < threshold:
                break

            direction = 'right' if diff > 0 else 'left'
            sign = 1 if direction == 'right' else -1

            # Spin wheels in opposite directions to rotate in place
            front_left_motor.setVelocity(3.0 * sign)
            rear_left_motor.setVelocity(3.0 * sign)
            front_right_motor.setVelocity(-3.0 * sign)
            rear_right_motor.setVelocity(-3.0 * sign)

        stop_motor()

    current_pos = get_map_position()

    distance = np.linalg.norm(map_target - current_pos)

    if distance < tolerance:
        print("[mvc_plan] Reached target!")
        stop_motor()
        return

    # Face the robot toward the target position
    desired_heading = angle_to_target(current_pos, map_target)
    rotate_to_heading(desired_heading)

    # Check and avoid obstacles in front
    distance_sensors_value = get_distances()
    if min(distance_sensors_value[0], distance_sensors_value[2]) < 0.3:
        adapt_direction(distance_sensors_value)

    # Move forward toward the target
    motor_speed = [8, 8]
    front_left_motor.setVelocity(motor_speed[0])
    front_right_motor.setVelocity(motor_speed[1])
    rear_left_motor.setVelocity(motor_speed[0])
    rear_right_motor.setVelocity(motor_speed[1])

def get_local_target():
    current_pos = get_position()
    map_x, map_y = convert_to_map_coordinates(current_pos[0], current_pos[1])
    r = 20

    intersect_list = []
    for x_border in [-r, r]:
        for j in range(-r, r+1):
            if grid_map[map_y+j, map_x+x_border] == 100:
                visual_map[map_y+j, map_x+x_border] = (0,0,255)
                intersect_list.append(np.array([map_x+x_border, map_y+j]))

    for y_border in [-r, r]:
        for i in range(-r, r+1):
            if grid_map[map_y+y_border, map_x+i] == 100:
                visual_map[map_y+y_border, map_x+i] = (0,0,255)
                intersect_list.append(np.array([map_x+i, map_y+y_border]))

    if len(intersect_list) == 0:
        print("No intersection found")
        return np.array([map_x+r, map_y+r])
    else:
        dist = [np.linalg.norm(intersect_point - np.array([map_x, map_y])) for intersect_point in intersect_list]
        print('--',intersect_list[np.argmin(dist)])
        return intersect_list[np.argmin(dist)]
    
def main():
    try:
        pcount = 0  
        while robot.step(TIME_STEP) != -1:

            target = get_local_target()
            print("Target: ", target)
            # grid_map[int(target[1])][int(target[0])] = 255
            draw_position_in_map()

            # distance_sensors_value = get_distances()
            # adapt_direction(distance_sensors_value)

            # motor_speed = [8, 8]

            # front_left_motor.setVelocity(motor_speed[0])
            # front_right_motor.setVelocity(motor_speed[1])
            # rear_left_motor.setVelocity(motor_speed[0])
            # rear_right_motor.setVelocity(motor_speed[1])

            map_resized = cv2.resize(visual_map, (500, 500))
            cv2.imshow("Occupancy Grid Map", map_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if pcount == 5:
                simple_plan(target)
                pcount = 0

            pcount += 1
            # time.sleep(0.5)

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(0)
        pass


main()