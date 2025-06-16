# tests/test_sample_simulation.py
import unittest
import sys
from pathlib import Path

# Add src directory to Python path to allow imports from simulations
# This assumes tests might be run from the root directory or tests directory
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from simulations.sample_simulation import SampleSimulation, run_simulation

class TestSampleSimulation(unittest.TestCase):

    def test_simulation_creation(self):
        """Test that SampleSimulation can be created."""
        sim = SampleSimulation()
        self.assertIsNotNone(sim)
        print("TestSampleSimulation: test_simulation_creation PASSED")

    def test_simulation_run(self):
        """Test that SampleSimulation's run method executes (placeholder)."""
        sim = SampleSimulation()
        # In a real scenario, you'd capture stdout or check for side effects.
        # For now, just ensure it runs without error.
        try:
            sim.run()
            print("TestSampleSimulation: test_simulation_run PASSED (no error)")
            ran_successfully = True
        except Exception as e:
            print(f"TestSampleSimulation: test_simulation_run FAILED with {e}")
            ran_successfully = False
        self.assertTrue(ran_successfully)

    def test_run_simulation_function(self):
        """Test the run_simulation function (placeholder)."""
        try:
            run_simulation() # Test with default params
            print("TestSampleSimulation: test_run_simulation_function (default params) PASSED")
            run_simulation(params={"test_param": 123}) # Test with custom params
            print("TestSampleSimulation: test_run_simulation_function (custom params) PASSED")
            ran_successfully = True
        except Exception as e:
            print(f"TestSampleSimulation: test_run_simulation_function FAILED with {e}")
            ran_successfully = False
        self.assertTrue(ran_successfully)

    def test_get_info(self):
        """Test the get_info method."""
        sim = SampleSimulation()
        self.assertEqual(sim.get_info(), "A basic sample simulation.")
        print("TestSampleSimulation: test_get_info PASSED")


if __name__ == '__main__':
    unittest.main()
