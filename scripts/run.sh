#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Navigate to the project root directory (assuming scripts/ is one level down from root)
PROJECT_ROOT="$SCRIPT_DIR/.."

# Path to the main Python script
MAIN_PY_PATH="$PROJECT_ROOT/src/main.py"

# Ensure the src directory is in PYTHONPATH if not running as a module
# This helps if main.py itself or simulations need to import other things from src
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Execute the main Python script, passing all arguments from this shell script
python3 "$MAIN_PY_PATH" "$@"
