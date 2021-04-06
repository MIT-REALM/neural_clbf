"""Very simple module to add the aerobench package to the path"""

import sys
import os


dirs = [
    "/home/cbd/src/mit/AeroBenchVVPython",
    "/home/cbd/src/AeroBenchVVPython",
    "/home/charles/src/AeroBenchVVPython",
]
for dir_name in dirs:
    if os.path.isdir(dir_name):
        sys.path.append(dir_name)
        break
