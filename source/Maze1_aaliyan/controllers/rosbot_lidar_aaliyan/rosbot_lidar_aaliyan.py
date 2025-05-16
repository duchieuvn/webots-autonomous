from controller import Robot
import math
import matplotlib.pyplot as plt
import random
import time

# --- Webots Setup ---
robot = Robot()
TIME_STEP = 64

# Motors
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

#distance sensors
front_right = robot.getDevice("fr_range")
front_left = robot.getDevice("fl_range")
rear_right = robot.getDevice("rr_range")
rear_left = robot.getDevice("rl_range")

# Enable them
for sensor in [front_right, front_left, rear_right, rear_left]:
    sensor.enable(TIME_STEP)
# Sensors
lidar = robot.getDevice("laser")
lidar.enable(TIME_STEP)
lidar.enablePointCloud()
fov = lidar.getFov()
num_rays = lidar.getHorizontalResolution()
lidar_range = lidar.getMaxRange()

front_left_encoder = robot.getDevice("fl_range")
front_right_encoder = robot.getDevice("fr_range")
rear_left_encoder = robot.getDevice("rl_range")
rear_right_encoder = robot.getDevice("rr_range")
front_left_encoder.enable(TIME_STEP)
front_right_encoder.enable(TIME_STEP)
rear_left_encoder.enable(TIME_STEP)
rear_right_encoder.enable(TIME_STEP)

# Robot parameters
wheel_radius = 0.033     # meters
axle_length = 0.16       # meters
lidar_range = 3.5        # meters (Webots default)

# Pose state
x, y, theta = 0.0, 0.0, 0.0
prev_front_left = front_left_encoder.getValue()
prev_front_right = front_right_encoder.getValue()
prev_rear_left = rear_left_encoder.getValue()
prev_rear_right = rear_right_encoder.getValue()
# Map store
map_points = []

# plt.ion()
# fig, ax = plt.subplots(figsize=(8, 8))
# sc = ax.scatter([], [], s=1)
# ax.set_title("Live Map")
# ax.set_xlabel("X (m)")
# ax.set_ylabel("Y (m)")
# ax.axis("equal")
# ax.grid(True)

last_turn_time = time.time()
stuck_start_time = time.time()
turning = False
turn_duration = 0.5  # seconds

# --- Main Loop ---
for _ in range(500):  # Run for 500 cycles
    if robot.step(TIME_STEP) == -1:
        break

    # Update odometry
    left = front_left_encoder.getValue()
    right = front_right_encoder.getValue()
    dl = wheel_radius * (left - prev_front_left)
    dr = wheel_radius * (right - prev_front_right)
    dc = (dl + dr) / 2.0
    dtheta = (dr - dl) / axle_length

    theta += dtheta
    x += dc * math.cos(theta)
    y += dc * math.sin(theta)

    prev_front_left = left
    prev_front_right = right

    # Get LiDAR scan
    ranges = lidar.getRangeImage()
    fov = lidar.getFov()
    num_rays = lidar.getHorizontalResolution()

    for i in range(num_rays):
        angle = -fov / 2 + i * (fov / num_rays)
        distance = ranges[i]

        if 0.05 < distance < lidar_range:
            # Local LiDAR coordinates
            lx = distance * math.cos(angle)
            ly = distance * math.sin(angle)

            # Global map coordinates
            gx = x + lx * math.cos(theta) - ly * math.sin(theta)
            gy = y + lx * math.sin(theta) + ly * math.cos(theta)

            map_points.append((gx, gy))
    # === Obstacle Avoidance ===
    ranges = lidar.getRangeImage()

# Slice the middle section of LiDAR to check front (±10°)
    mid_range = int(num_rays / 2)
    window = 20  # Number of rays to check around the center
    front_ranges = ranges[mid_range - window: mid_range + window]

# Compute minimum distance in front
    min_front = min(front_ranges)

# Set motion based on obstacle distance
    ranges = lidar.getRangeImage()
    mid = int(num_rays / 2)
    window = 10

# Front scan window
    front_ranges = ranges[mid - window : mid + window]
    left_ranges = ranges[:mid - window]
    right_ranges = ranges[mid + window:]

    min_front = min(front_ranges)
    avg_left = sum(left_ranges) / len(left_ranges)
    avg_right = sum(right_ranges) / len(right_ranges)

    current_time = time.time()
    
# Read sensor values
    # Read distance sensor values
    fr = front_right.getValue()
    fl = front_left.getValue()
    rr = rear_right.getValue()
    rl = rear_left.getValue()

## Obstacle detection threshold (raw values; tune as needed)
    OBSTACLE = 0.3

# Determine if each side is blocked
    front_blocked = fl < OBSTACLE or fr < OBSTACLE
    left_blocked = fl < OBSTACLE or rl < OBSTACLE
    right_blocked = fr < OBSTACLE or rr < OBSTACLE
    rear_blocked = rl < OBSTACLE or rr < OBSTACLE

# === Decision Making ===
    if not front_blocked:
        print("Move Forward")
        front_left_motor.setVelocity(2.0)
        front_right_motor.setVelocity(2.0)
        rear_left_motor.setVelocity(2.0)
        rear_right_motor.setVelocity(2.0)

    elif not left_blocked:
            print("Turning Leftt")
            front_left_motor.setVelocity(-2.0)
            front_right_motor.setVelocity(2.0)
            rear_left_motor.setVelocity(-2.0)
            rear_right_motor.setVelocity(2.0)
    elif not right_blocked:
            print("Turning Right")
            front_left_motor.setVelocity(2.0)
            front_right_motor.setVelocity(-2.0)
            rear_left_motor.setVelocity(2.0)
            rear_right_motor.setVelocity(-2.0)
    elif not rear_blocked:
            print("Backing up")
            front_left_motor.setVelocity(-2.0)
            front_right_motor.setVelocity(-2.0)
            rear_left_motor.setVelocity(-2.0)
            rear_right_motor.setVelocity(-2.0)
    else:
        print("Spinning in Place")
        front_left_motor.setVelocity(-1.5)
        front_right_motor.setVelocity(1.5)
        rear_left_motor.setVelocity(-1.5)
        rear_right_motor.setVelocity(1.5)
# === If obstacle ahead === 
    #if min_front < 0.4:
      #  turning = True
       # stuck_start_time = time.time()

    # Smart turning: turn toward more open space
      #  if avg_left > avg_right:
       #     print("Turning left",avg_left,avg_right)
        #    front_left_motor.setVelocity(-2.0)
         #   rear_left_motor.setVelocity(-2.0)
         #   front_right_motor.setVelocity(2.0)
           # rear_right_motor.setVelocity(2.0)
      #  else:
      #      print("Turning right",avg_left,avg_right)
         #   front_left_motor.setVelocity(2.0)
         #   front_right_motor.setVelocity(-2.0)
          #  rear_left_motor.setVelocity(2.0)
           # rear_right_motor.setVelocity(-2.0)
      #  last_turn_time = current_time

# === If stuck for too long (e.g., spinning) ===
   # elif turning and (current_time - last_turn_time > turn_duration):
      #  turning = False

# === If not blocked, drive forward ===
    # elif not turning:
    # Detect if stuck
        # if current_time - stuck_start_time > 5.0:  # 5 seconds stuck
            # print("Stuck! Doing random turn.")
            # turning = True
            # stuck_start_time = current_time
            # last_turn_time = current_time

            # rand_dir = random.choice(['left', 'right'])
            # if rand_dir == 'left':
                # front_left_motor.setVelocity(-2.0)
                # front_right_motor.setVelocity(2.0)
                # rear_left_motor.setVelocity(-2.0)
                # rear_right_motor.setVelocity(2.0)
            # else:
                # front_left_motor.setVelocity(2.0)
                # front_right_motor.setVelocity(-2.0)
                # rear_left_motor.setVelocity(2.0)
                # rear_right_motor.setVelocity(-2.0)
   # else:
    #    front_left_motor.setVelocity(2.0)
    #    front_right_motor.setVelocity(2.0)
     #   rear_left_motor.setVelocity(2.0)
     #   rear_right_motor.setVelocity(2.0)



# --- Plot the Map ---
    # Add new points from current scan
        # Add new points from current scan
    # scan_points = []

    # for i in range(num_rays):
        # angle = -fov / 2 + i * (fov / num_rays)
        # distance = ranges[i]

        # if 0.05 < distance < lidar_range:
            # lx = distance * math.cos(angle)
            # ly = distance * math.sin(angle)

            # gx = x + lx * math.cos(theta) - ly * math.sin(theta)
            # gy = y + lx * math.sin(theta) + ly * math.cos(theta)

            # scan_points.append((gx, gy))

    # if scan_points:
        # map_points.extend(scan_points)
        # x_coords, y_coords = zip(*map_points)
        # sc.set_offsets(list(zip(x_coords, y_coords)))
        # plt.pause(0.001)


# plt.ioff()
# plt.show()

