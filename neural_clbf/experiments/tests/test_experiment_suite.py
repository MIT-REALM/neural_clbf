import pandas as pd

from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.datamodules.episodic_datamodule import EpisodicDataModule
from neural_clbf.experiments import ExperimentSuite
from neural_clbf.experiments.tests.mock_experiment import MockExperiment
from neural_clbf.systems.tests.mock_system import MockSystem


def test_experiment_suite():
    # Define a mock experiment to use in the suite
    experiment_1 = MockExperiment("mock_experiment_1")
    experiment_2 = MockExperiment("mock_experiment_2")

    # Create the suite
    experiment_suite = ExperimentSuite([experiment_1, experiment_2])

    # Define the model system
    params = {}
    system = MockSystem(params)
    # Define the datamodule
    initial_domain = [
        (-1.0, 1.0),
        (-1.0, 1.0),
    ]
    dm = EpisodicDataModule(system, initial_domain)

    # Instantiate with a list of only one scenarios
    scenarios = [params]
    controller = NeuralCLBFController(system, scenarios, dm, experiment_suite)

    # Test running the experiments
    results = experiment_suite.run_all(controller)

    # The results should be a list of DataFrames
    assert isinstance(results, list)
    assert len(results) == 2
    assert isinstance(results[0], pd.DataFrame)

    # Test plotting
    fig_handles = experiment_suite.run_all_and_plot(controller, display_plots=False)
    # fig_handles should be a list of tuples (name, figure)
    assert isinstance(fig_handles, list)
    assert len(fig_handles) == 2
    assert isinstance(fig_handles[0], tuple)
    assert len(fig_handles[0]) == 2
