import pygame
from controller import Robot, Keyboard, Supervisor
from my_robot import MyRobot
from visualization import Visualizer
import time

def main():
    robot = MyRobot()
    # vis = Visualizer()

    running = True
    count = 0
    red_wall_detected = False

    while running and robot.step(robot.time_step) != -1:
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         running = False

        # if not red_wall_detected and robot.there_is_red_wall():
        distances = robot.get_distances()
        if min(distances[0], distances[2]) < 0.4 and robot.there_is_red_wall():
            # red_wall_detected = True
            # if min(distances[0], distances[2]) < 1:
            print("Red wall detected! Rotating 180 degrees...", flush=True)
            robot.stop_motor()
            robot.turn_180_degrees()
            time.sleep(1)
        else:
            robot.adapt_direction()
            robot.set_robot_velocity(6, 6)

        # vis.clear_screen()
        # vis.update_screen_with_map(robot.grid_map)
        # vis.draw_robot(robot.get_map_position())
        # vis.display_screen()

        count += 1

main()
