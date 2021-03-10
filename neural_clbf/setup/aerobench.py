"""Very simple module to add the aerobench package to the path"""

import sys
import os


laptop_dir = "/home/cbd/src/mit/AeroBenchVVPython"
server_dir = "/home/cbd/src/AeroBenchVVPython"
if os.path.isdir(laptop_dir):
    sys.path.append(laptop_dir)
else:
    sys.path.append(server_dir)
