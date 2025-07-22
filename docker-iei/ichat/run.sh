#!/bin/bash

# Get the directory where the script is located.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# CD to the parent directory of the script's location (/app),
# which is the correct execution context for running ichat as a module.
cd "$SCRIPT_DIR/.."

# This script runs the ichat application as a Python module,
# which is necessary to handle relative imports correctly within the package.
# The "$@" allows passing all command-line arguments to the serve script.
python3 -m ichat.serve "$@" 