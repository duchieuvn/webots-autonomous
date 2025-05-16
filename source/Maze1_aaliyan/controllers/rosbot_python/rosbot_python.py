from controller import Robot, Motor, PositionSensor, Camera, RangeFinder, Lidar, Accelerometer, Gyro, Compass, DistanceSensor, 
import math
import matplotlib.pyplot as plt

TIME_STEP = 32
MAX_VELOCITY = 26
BASE_SPEED = 6.0

# Set coefficients for collision avoidance
coefficients = [[15.0, -9.0], [-15.0, 9.0]]

# Initialize robot
robot = Robot()

# Motors
front_left_motor = robot.getDevice("fl_wheel_joint")
front_right_motor = robot.getDevice("fr_wheel_joint")
rear_left_motor = robot.getDevice("rl_wheel_joint")
rear_right_motor = robot.getDevice("rr_wheel_joint")

for motor in [front_left_motor, front_right_motor, rear_left_motor, rear_right_motor]:
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)

# Position sensors
fl_sensor = robot.getDevice("front left wheel motor sensor")
fr_sensor = robot.getDevice("front right wheel motor sensor")
rl_sensor = robot.getDevice("rear left wheel motor sensor")
rr_sensor = robot.getDevice("rear right wheel motor sensor")

for sensor in [fl_sensor, fr_sensor, rl_sensor, rr_sensor]:
    sensor.enable(TIME_STEP)

wheel_radius = 0.033  # Check your ROSbot wheel radius
axle_length = 0.16     # Distance between the wheels
lidar_range = 3.5 

prev_left = left_position_sensor.getValue()
prev_right = right_position_sensor.getValue()

x, y, theta = 0.0, 0.0, 0.0  # Initial pose
map_points = []
# RGBD camera
camera_rgb = robot.getDevice("camera rgb")
camera_depth = robot.getDevice("camera depth")
camera_rgb.enable(TIME_STEP)
camera_depth.enable(TIME_STEP)

# LIDAR
lidar = robot.getDevice("laser")
lidar.enable(TIME_STEP)
lidar.enablePointCloud()

# IMU sensors
accelerometer = robot.getDevice("imu accelerometer")
gyro = robot.getDevice("imu gyro")
compass = robot.getDevice("imu compass")
accelerometer.enable(TIME_STEP)
gyro.enable(TIME_STEP)
compass.enable(TIME_STEP)

# Distance sensors
distance_sensors = [
    robot.getDevice("fl_range"),
    robot.getDevice("rl_range"),
    robot.getDevice("fr_range"),
    robot.getDevice("rr_range"),
]

for ds in distance_sensors:
    ds.enable(TIME_STEP)

# Main loop
while robot.step(TIME_STEP) != -1:
    # Read accelerometer
    a = accelerometer.getValues()
    print(f"accelerometer values = {a[0]:.2f} {a[1]:.2f} {a[2]:.2f}")

    # Read distance sensors
    ds_values = [ds.getValue() for ds in distance_sensors]

    #Read Lidar
    ranges = lidar.getRangeImage()
    number_of_points = lidar.getHorizontalResolution()
    fov = lidar.getFov()

    # Optional: Convert polar to cartesian for mapping
    for i in range(number_of_points):
        angle = -fov/2 + i * (fov / number_of_points)
        distance = ranges[i]
        x = distance * math.cos(angle)
        y = distance * math.sin(angle)
    
    # Compute motor speeds
    avoidance_speed = [0.0, 0.0]
    motor_speed = [0.0, 0.0]

    for i in range(2):
        for j in range(1, 3):  # Only using rl_range and fr_range
            diff = 2.0 - ds_values[j]
            avoidance_speed[i] += diff * diff * coefficients[i][j - 1]
        motor_speed[i] = BASE_SPEED + avoidance_speed[i]
        motor_speed[i] = min(motor_speed[i], MAX_VELOCITY)

    # Set motor speeds
    front_left_motor.setVelocity(motor_speed[0])
    rear_left_motor.setVelocity(motor_speed[0])
    front_right_motor.setVelocity(motor_speed[1])
    rear_right_motor.setVelocity(motor_speed[1])
    
    #Get and update pose
    left = left_position_sensor.getValue()
    right = right_position_sensor.getValue()

    dl = wheel_radius * (left - prev_left)
    dr = wheel_radius * (right - prev_right)
    dc = (dl + dr) / 2.0
    dtheta = (dr - dl) / axle_length

    # Update pose
    theta += dtheta
    x += dc * math.cos(theta)
    y += dc * math.sin(theta)

    prev_left = left
    prev_right = right
