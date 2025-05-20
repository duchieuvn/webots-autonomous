from controller import Robot, Keyboard, Supervisor

robot = Supervisor()

def main():
    grid_map, start_point, end_point = robot.explore()
    path = robot.find_path(start_point, end_point)
    robot.path_following_pipeline(path)

main()
