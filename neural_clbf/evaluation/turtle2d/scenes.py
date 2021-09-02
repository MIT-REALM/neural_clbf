from shapely.geometry import box
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon

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


def saved_random_scene():
    # This is kind of like "return 4  # chosen by fair dice roll"
    # I've made one random scene (selected so that the goal is clear) and
    # saved the details here

    scene = Scene([])
    room_size = 10.0
    scene.add_walls(room_size)

    scene.obstacles.append(
        Polygon(
            [
                (-4.2136676463868366, 0.5679646499153548),
                (-2.743330656989159, -0.30710755158835323),
                (-2.0696517186228496, 0.8248391156192505),
                (-3.5399887080205277, 1.6999113171229587),
                (-4.2136676463868366, 0.5679646499153548),
            ]
        )
    )
    scene.obstacles.append(
        Polygon(
            [
                (0.2810067281373088, 1.8313846995884488),
                (-1.1178221533074106, 1.1851621034840258),
                (-0.46066806204718935, -0.2373293537255825),
                (0.9381608193975304, 0.4088932423788404),
                (0.2810067281373088, 1.8313846995884488),
            ]
        )
    )
    scene.obstacles.append(
        Polygon(
            [
                (3.0150536444465867, -3.6712772807033898),
                (4.4733962529696205, -3.4652282279839715),
                (4.31357901681553, -2.3340981306412214),
                (2.8552364082924964, -2.54014718336064),
                (3.0150536444465867, -3.6712772807033898),
            ]
        )
    )
    scene.obstacles.append(
        Polygon(
            [
                (1.9915836749108597, 3.696365684856654),
                (1.1906921081176052, 4.626324483349467),
                (0.05418969918334149, 3.6475551051459822),
                (0.8550812659765963, 2.7175963066531694),
                (1.9915836749108597, 3.696365684856654),
            ]
        )
    )
    scene.obstacles.append(
        Polygon(
            [
                (-0.8220082932121368, 1.2901466027527497),
                (-2.0028796561132642, 0.6294746466154549),
                (-1.2854031624122628, -0.6529279695327568),
                (-0.10453179951113567, 0.007743986604538122),
                (-0.8220082932121368, 1.2901466027527497),
            ]
        )
    )
    scene.obstacles.append(
        Polygon(
            [
                (3.607829125311771, 0.5473325823272353),
                (4.215626943185372, 2.0756404349101327),
                (2.902897580988146, 2.597704128068316),
                (2.295099763114544, 1.0693962754854187),
                (3.607829125311771, 0.5473325823272353),
            ]
        )
    )
    scene.obstacles.append(
        Polygon(
            [
                (-1.7080890714472434, -1.985992050420569),
                (-0.5808673151891823, -1.7964378971591917),
                (-0.7793064920166237, -0.616379567836322),
                (-1.9065282482746848, -0.8059337210976993),
                (-1.7080890714472434, -1.985992050420569),
            ]
        )
    )
    scene.obstacles.append(
        Polygon(
            [
                (3.0496394871572035, -1.6500303672785277),
                (2.1257656941885603, -3.1212508648447477),
                (3.3240219557738753, -3.873712891556976),
                (4.247895748742518, -2.402492393990756),
                (3.0496394871572035, -1.6500303672785277),
            ]
        )
    )

    return scene
