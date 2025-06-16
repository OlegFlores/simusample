# src/simulations/sample_simulation.py

class SampleSimulation:
    def __init__(self, params=None):
        self.params = params if params else {}
        print(f"Initializing SampleSimulation with parameters: {self.params}")

    def run(self):
        print("Running SampleSimulation...")
        # Placeholder for actual simulation logic
        print("SampleSimulation finished.")

    @staticmethod
    def get_info():
        return "A basic sample simulation."

def run_simulation(params=None):
    sim = SampleSimulation(params)
    sim.run()
