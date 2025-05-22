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
    # grid_map, start_point, end_point = robot.explore()
    # path = robot.find_path(start_point, end_point)
    # robot.path_following_pipeline(path)
    
    running = True
    count = 0
    while robot.step(robot.time_step) != -1 and running:
        for event in pygame.event.get():  # Xử lý tất cả sự kiện
            if event.type == pygame.QUIT:  # Nếu người dùng đóng cửa sổ
                running = False
        vis.clear_screen()

        robot.adapt_direction()
        robot.set_robot_velocity(8,8)
        points = robot.get_pointcloud_world_coordinates()
        map_points = robot.convert_to_map_coordinate_matrix(points)
        
        if count % 20 == 0 and not robot.is_turning():
            for map_point in map_points:
                robot.draw_bresenham_line(map_point)
            #    vis.draw_line(cur_map_pos, map_point)

        vis.update_screen_with_map(robot.grid_map)
        vis.draw_robot(robot)
        vis.display_screen()

        count += 1


main()  

