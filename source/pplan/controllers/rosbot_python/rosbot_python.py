import pygame
from controller import Robot, Keyboard, Supervisor
from setup import setup_robot
from my_robot import MyRobot
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2
from visualization import Visualizer 

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

def generate_path():
    start_point = [200, 250]
    path = []
    end_x = 400
    end_y = 400
    for i in range(3):
        random_int = random.randint(200, 350)
        if i % 2 == 0:
            end_x = start_point[0] + random_int
        else:
            end_y = start_point[1] + random_int

        path += generate_random_curved_path(start_point, (end_x, end_y))
        start_point = [end_x + 30, end_y + 30]

    return path

def main():
    vis = Visualizer()
    robot = MyRobot()    
    path = generate_path()
    target_index = 0
    target = path[target_index]

    while robot.step() != -1:
        vis.visualize_game(robot, path, target)

        if robot.dwa_plan(target):
            target_index += 1
            if target_index < len(path):
                target = path[target_index]
            else:
                break

main()
