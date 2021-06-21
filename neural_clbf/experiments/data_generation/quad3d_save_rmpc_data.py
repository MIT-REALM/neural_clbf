from neural_clbf.systems import Quad3D
from neural_clbf.controllers.comparisons.robust_mpc_controller import (
    Quad3DRobustMPCController,
)

from neural_clbf.experiments.data_generation.quad3d_rollout import (
    quad3d_rollout,
)


def doMain():
    valid_params = {
        "m": 1.0,
    }
    quad3d = Quad3D(valid_params)

    controller_period = 0.1
    rmpc_controller = Quad3DRobustMPCController(quad3d, controller_period)

    n_sims = 1
    goal_errs_1 = []
    failures_1 = 0
    for i in range(n_sims):
        goal_err, failure = quad3d_rollout(
            rmpc_controller, "rMPC", controller_period, quad3d, save=False
        )
        # print(goal_err)
        # print(failure)
        goal_errs_1.append(goal_err)
        if failure:
            failures_1 += 1

        # with open("sim_traces/quad3d_mpc_param_sweep.txt", 'a') as file1:
        #     file1.write(f"{goal_err}, {failure}")

    # controller_period = 0.25
    # rmpc_controller = Quad3DRobustMPCController(quad3d, controller_period)

    # n_sims = 100
    # goal_errs_25 = []
    # failures_25 = 0
    # for i in range(n_sims):
    #     goal_err, failure = quad3d_rollout(
    #         rmpc_controller, "rMPC", controller_period, quad3d, save=False
    #     )
    #     # print(goal_err)
    #     # print(failure)
    #     goal_errs_25.append(goal_err)
    #     if failure:
    #         failures_25 += 1

    # print("===================================")
    # print("dt = 0.25")
    # print("Failures_25:")
    # print(failures_25)
    # print(goal_errs_25)
    # print("Average goal error")
    # print(sum(goal_errs_25) / n_sims)


if __name__ == "__main__":
    doMain()
