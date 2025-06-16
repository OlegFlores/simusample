# src/main.py
import argparse
import importlib
import pkgutil
import sys
from pathlib import Path

# Add src directory to Python path to allow direct imports from simulations
sys.path.insert(0, str(Path(__file__).resolve().parent))

import simulations

def list_simulations():
    print("Available simulations:")
    for _, name, _ in pkgutil.iter_modules(simulations.__path__):
        try:
            module = importlib.import_module(f"simulations.{name}")
            if hasattr(module, 'get_info') and callable(module.get_info):
                print(f"  - {name}: {module.get_info()}")
            elif hasattr(module, 'SampleSimulation') and hasattr(module.SampleSimulation, 'get_info'):
                 # Fallback for the sample if get_info is a class method
                sim_class = getattr(module, 'SampleSimulation')
                if callable(sim_class.get_info):
                     print(f"  - {name}: {sim_class().get_info()}")
                else:
                    print(f"  - {name}: (No description provided)")
            else:
                print(f"  - {name}: (No description provided)")
        except ImportError:
            print(f"  - {name}: (Error importing)")


def main():
    parser = argparse.ArgumentParser(description="Run physical world simulations.")
    parser.add_argument(
        "simulation_name",
        nargs="?", # Makes the argument optional
        help="The name of the simulation to run (e.g., sample_simulation). Lists available simulations if not provided.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available simulations."
    )
    parser.add_argument(
        "--params",
        type=str,
        help="JSON string of parameters to pass to the simulation (e.g., '{\"speed\": 2.0, \"mass\": 5}')",
    )

    args = parser.parse_args()

    if args.list or not args.simulation_name:
        list_simulations()
        return

    simulation_name = args.simulation_name
    params_dict = {}
    if args.params:
        import json
        try:
            params_dict = json.loads(args.params)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON string for --params: {args.params}")
            return

    try:
        # Dynamically import the simulation module
        module_path = f"simulations.{simulation_name}"
        simulation_module = importlib.import_module(module_path)

        # Check for a 'run_simulation' function or a specific class like 'SampleSimulation'
        if hasattr(simulation_module, "run_simulation") and callable(simulation_module.run_simulation):
            simulation_module.run_simulation(params=params_dict)
        elif hasattr(simulation_module, "SampleSimulation"): # Fallback for the initial sample
            sim_instance = simulation_module.SampleSimulation(params=params_dict)
            sim_instance.run()
        else:
            print(f"Error: Could not find a 'run_simulation' function or 'SampleSimulation' class in '{simulation_name}'.")
            list_simulations()

    except ImportError:
        print(f"Error: Simulation '{simulation_name}' not found.")
        list_simulations()
    except Exception as e:
        print(f"An error occurred while running simulation '{simulation_name}': {e}")

if __name__ == "__main__":
    main()
