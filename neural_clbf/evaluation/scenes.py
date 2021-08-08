from shapely.geometry import box
from shapely.affinity import rotate, translate

from neural_clbf.systems.planar_lidar_system import Scene


def add_to_scene(scene, obstacles):
    """Add each obstacle in obstacles to the scene"""
    for obstacle in obstacles:
        scene.add_obstacle(obstacle)


def bugtrap():
    """Return a list of obstacles for a simple bugtrap"""
    scene = Scene([])
    scene.add_walls(10.0)

    obstacles = []
    obstacles.append(box(-1.1, -1.0, -0.9, 1.0))  # vertical wall
    obstacles.append(box(-2.0, 1.0, -0.9, 1.2))  # upper horizontal
    obstacles.append(box(-2.0, -1.2, -0.9, -1.0))  # lower horizontal

    add_to_scene(scene, obstacles)

    return scene


def room_4():
    """A house with 4 rooms connected by doors and with some furniture in the way"""
    scene = Scene([])

    obstacles = []
    # Make the outer walls
    obstacles.append(box(-10.25, -10.25, 10.25, -10.0))
    obstacles.append(box(-10.25, 10.0, 10.25, 10.25))
    obstacles.append(box(-10.25, -10.25, -10.0, 10.25))
    obstacles.append(box(10.0, -10.25, 10.25, 10.25))

    # Make the inner walls
    obstacles.append(box(-0.25, -5.0, 0.25, 7.0))  # central wall
    obstacles.append(box(-0.25, 9.0, 0.25, 10.0))  # upper central wall
    obstacles.append(box(-0.25, -10.0, 0.25, -8.0))  # lower central wall

    obstacles.append(box(-8.0, -1.5, -0.25, -1.0))  # left inner wall
    obstacles.append(box(0.25, 1.0, 4.0, 1.5))  # right inner wall 1
    obstacles.append(box(6.0, 1.0, 10.0, 1.5))  # right inner wall 2

    # Make the furniture
    chair = box(-0.75, -0.75, 0.75, 0.75)
    obstacles.append(translate(rotate(chair, 45), -8.0, -5.0))
    obstacles.append(translate(rotate(chair, 75), -5.0, -8.0))
    obstacles.append(translate(rotate(chair, 75), -5.0, -3.0))
    obstacles.append(translate(rotate(chair, 15), -2.0, -7.0))
    table = box(-1.0, -2.5, 1.0, 2.5)
    obstacles.append(translate(table, -5.0, 5.0))
    obstacles.append(translate(chair, -6.8, 6.0))
    obstacles.append(translate(chair, -6.8, 4.0))
    obstacles.append(translate(chair, -3.2, 6.0))
    obstacles.append(translate(chair, -3.2, 4.0))

    # Add a maze to one room
    obstacles.append(box(0.25, -8.2, 8.0, -8.0))
    obstacles.append(box(2.0, -5.0, 10.0, -4.8))
    obstacles.append(box(0.25, -2.2, 8.0, -2))

    # Hide the goal in a box
    obstacles.append(box(3.5, 4.0, 6.5, 4.2))
    obstacles.append(box(3.5, 6.5, 5.5, 6.7))
    obstacles.append(box(3.5, 4.2, 3.7, 6.5))
    obstacles.append(box(6.3, 4.2, 6.5, 5.5))

    # Make the goal at 0, 0 by translating
    translated_obstacles = []
    for obstacle in obstacles:
        translated_obstacles.append(translate(obstacle, -5.0, -5.0))

    add_to_scene(scene, translated_obstacles)

    return scene
