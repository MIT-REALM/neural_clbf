np = py.importlib.import_module("numpy");

x = np.zeros([int32(1), int32(6)]);
u_ref = np.zeros([int32(1), int32(3)]);
relaxation_penalty = 1000.0;
py.matlab_interface.linear_satellite_cbf_qp_filter(x, u_ref, relaxation_penalty)