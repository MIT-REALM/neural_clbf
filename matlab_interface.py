import neural_clbf.evaluation.matlab_interface.linear_satellite_cbf as lin_sat_cbf


def linear_satellite_cbf_qp_filter(x, u_ref, relaxation_penalty):
    return lin_sat_cbf.cbf_qp_filter(x, u_ref, relaxation_penalty)
