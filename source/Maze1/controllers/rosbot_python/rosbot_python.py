import pygame
from controller import Robot, Keyboard, Supervisor
from setup import setup_robot
from my_robot import MyRobot

def main():
    robot = MyRobot()

    grid_map, start_point, end_point = robot.explore()
    path = robot.find_path(start_point, end_point)
    robot.path_following_pipeline(path)

main()
