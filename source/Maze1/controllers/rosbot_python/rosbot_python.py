import pygame
from controller import Robot, Keyboard, Supervisor
from setup import setup_robot
from my_robot import MyRobot
from visualization import Visualizer

import numpy as np

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
    vis = Visualizer()
    grid_map, start_point, end_point = robot.explore()
    # affected_map = robot.inflate_obstacles(grid_map)

    path = robot.find_path(start_point, end_point)
    print("Path found:", path)

    # Code-block to test find_path function
    count = 0
    while robot.step(robot.time_step) != -1 and count < 1000:
        for event in pygame.event.get(): 
            if event.type == pygame.QUIT:
                running = False
        vis.clear_screen()
    
        vis.update_screen_with_map(grid_map)
        vis.draw_point(start_point, (0, 255, 0))  # Start point in green
        vis.draw_point(end_point, (255, 0, 0))    # End point in red
        # vis.draw_path(path)
        vis.draw_robot(robot.get_map_position())
        vis.display_screen()
        count += 1

    # robot.path_following_pipeline(path)
    
main()  

