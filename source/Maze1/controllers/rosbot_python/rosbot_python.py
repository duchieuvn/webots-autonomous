import pygame
from controller import Robot, Keyboard, Supervisor
from setup import setup_robot
from my_robot import MyRobot
from visualization import Visualizer
import random
import numpy as np
import cv2
import math

def draw_bresenham_line(grid_map, start, end):
    x1, y1 = start
    x2, y2 = end
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        grid_map[y1][x1] = 0  # Mark the line on the grid map
        if x1 == x2 and y1 == y2:
            break
        err2 = err * 2
        if err2 > -dy:
            err -= dy
            x1 += sx
        if err2 < dx:
            err += dx
            y1 += sy    

def main():
    robot = MyRobot()
    # vis = Visualizer()
    # grid_map, start_point, end_point = robot.explore()
    # affected_map = robot.inflate_obstacles(grid_map)
    robot.grid_map= robot.generate_maze_grid_map()
    robot.visualize_grid_map(robot.grid_map)
    start_point = [200, 250]
    end_point = [600, 500]
    path = robot.find_path(start_point, end_point)
    robot.visualize_path_cv2(path)
    robot.path_following_pipeline(path)

main()  