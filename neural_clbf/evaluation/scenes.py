from shapely.geometry import box
from shapely.affinity import rotate

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
    obstacles.append(box(-1.1, -1.0, -0.9, 1.0))
    obstacles.append(box(-2.0, 1.0, -0.9, 1.2))
    obstacles.append(box(-2.0, -1.2, -0.9, -1.0))

    add_to_scene(scene, obstacles)

    return scene


def room_4():
    """A house with 4 rooms connected by doors and with some furniture in the way"""
    scene = Scene([])
    scene.add_walls(20.0)
