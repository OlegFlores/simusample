# Python Physical World Simulations Project

This project provides a skeleton for running various physical world simulations.
You can select which simulation to run from the command line.

## Project Structure

-   `src/`: Contains the main application code.
    -   `main.py`: The entry point of the application. It handles command-line arguments to select and run simulations.
    -   `simulations/`: Directory where individual simulation modules are stored.
        -   `__init__.py`: Makes `simulations` a Python package.
        -   `sample_simulation.py`: An example simulation. Add your new simulations here.
-   `tests/`: Contains unit tests for the simulations.
    -   `test_sample_simulation.py`: An example test file.
-   `scripts/`: Contains helper scripts.
    -   `run.sh`: A script to easily run the simulations.
-   `requirements.txt`: Lists the Python dependencies for this project.
-   `README.md`: This file.

## Setup Instructions

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a Python virtual environment:**
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python3 -m venv venv
    ```

3.  **Activate the virtual environment:**
    -   On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```
    -   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    You should see `(venv)` at the beginning of your command prompt.

4.  **Install dependencies:**
    Install all required libraries using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## Running Simulations

You can run simulations using the `scripts/run.sh` script.

1.  **Make sure the script is executable (if you cloned fresh or it lost permissions):**
    ```bash
    chmod +x scripts/run.sh
    ```

2.  **List available simulations:**
    To see which simulations are available, run:
    ```bash
    ./scripts/run.sh --list
    ```
    or
    ```bash
    ./scripts/run.sh
    ```

3.  **Run a specific simulation:**
    Use the name of the simulation file (without the `.py` extension) as an argument.
    For example, to run the `sample_simulation`:
    ```bash
    ./scripts/run.sh sample_simulation
    ```

4.  **Run a simulation with parameters:**
    Some simulations might accept parameters. You can pass them as a JSON string.
    For example:
    ```bash
    ./scripts/run.sh sample_simulation --params '{"speed": 2.5, "mass": 10}'
    ```
    The `sample_simulation` currently just prints these parameters.

## Adding a New Simulation

1.  **Create a new Python file** in the `src/simulations/` directory (e.g., `my_new_simulation.py`).
2.  **Implement your simulation logic** in this file. It's recommended to have:
    *   A main class for your simulation (e.g., `MyNewSimulation`).
    *   A `run_simulation(params=None)` function that initializes and runs your simulation. This is what `main.py` will try to call.
    *   Optionally, a `get_info()` function at the module level that returns a brief string description of the simulation. This description will be shown when listing simulations.
        ```python
        # src/simulations/my_new_simulation.py

        def get_info():
            return "This is my new awesome simulation about X."

        class MyNewSimulation:
            def __init__(self, params=None):
                self.params = params if params else {}
                print(f"Initializing MyNewSimulation with parameters: {self.params}")
                # Your initialization logic here

            def run(self):
                print(f"Running MyNewSimulation with {self.params}...")
                # Your simulation logic here
                print("MyNewSimulation finished.")

        def run_simulation(params=None):
            sim = MyNewSimulation(params)
            sim.run()

        # To make it runnable directly (optional)
        if __name__ == '__main__':
            run_simulation({"example_param": "value"})
        ```
3.  **Add tests** for your new simulation in the `tests/` directory (e.g., `test_my_new_simulation.py`).
    ```python
    # tests/test_my_new_simulation.py
    import unittest
    import sys
    from pathlib import Path

    # Adjust path to import from src
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "src"))

    from simulations.my_new_simulation import MyNewSimulation, run_simulation # Or whatever you named it

    class TestMyNewSimulation(unittest.TestCase):
        def test_creation(self):
            sim = MyNewSimulation()
            self.assertIsNotNone(sim)

        def test_run(self):
            # Add more meaningful assertions here
            run_simulation()
            self.assertTrue(True) # Placeholder

    if __name__ == '__main__':
        unittest.main()
    ```

4.  The `main.py` script will automatically discover any new `*.py` files in the `src/simulations/` directory and list them if they can be imported and (optionally) have a `get_info()` function.

## Running Tests

To run all tests, navigate to the project root directory and run:
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```
Make sure your virtual environment is activated to ensure all dependencies are available.
