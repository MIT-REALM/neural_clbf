% The interface expects the state and control to be numpy arrays
np = py.importlib.import_module("numpy");

% Here's an example of making some numpy arrays
x = np.array([1, 0, 0, 0, 0, 0]);
u_ref = np.array([0, 0, 0]);

% The system also expects a relaxation penalty, which is the cost
% assigned to relaxing the CBF conditions. This shouldn't happen inside the
% safe set, but it might happen outside of it.
relaxation_penalty = 1000.0;

% Call the interface to get the result. The result is a tuple, where
% the first element is the filtered control and the second element
% is the relaxations required to make the QP feasible. If x and u_ref have
% N rows, then u_filtered will be N x 3 and relaxations will be N x 6 (if
% all elements of relaxations are small, < 1e-5, then the QP was feasible).
result = py.matlab_interface.linear_satellite_cbf_qp_filter(x, u_ref, relaxation_penalty);
u_filtered = result{1}.double;
relaxations = result{2}.double;