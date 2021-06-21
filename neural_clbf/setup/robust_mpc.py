"""Very simple module to find the MATLAB directory for robust MPC"""
import os


possible_paths = [
    "C:\\Users\\qinzy\\Documents\\MATLAB\\robust_mpc",
    "/home/cbd/src/mit/robust_mpc",
]
robust_mpc_path = ""

for dir_name in possible_paths:
    if os.path.isdir(dir_name):
        robust_mpc_path = dir_name
        break
