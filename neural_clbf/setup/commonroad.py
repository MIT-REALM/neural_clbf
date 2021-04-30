"""Very simple module to add the commonroad vehicle models package to the path"""

import sys
import os


dirs = [
    "/home/cbd/src/mit/commonroad-vehicle-models/PYTHON",
    "/home/cbd/src/commonroad-vehicle-models/PYTHON",
    "/home/charles/src/commonroad-vehicle-models/PYTHON",
]
for dir_name in dirs:
    if os.path.isdir(dir_name):
        sys.path.append(dir_name)
        break
