#!/bin/bash

# This script installs wheel packages from a specified directory.
# It checks if a package is already installed before attempting installation.
# If the package is found in the environment, it will be skipped.
# The script also supports listing packages to be installed or cleaning up
# .whl files for packages that are already installed.

# --- Script Usage ---
usage() {
    echo "Usage: $0 -p <path> [-l | -d] [-h]"
    echo ""
    echo "Manages and installs wheel packages from a specified directory."
    echo "Default action is to install packages that are not already present in the environment."
    echo ""
    echo "Options:"
    echo "  -p <path>   (Required) Path to the directory containing .whl files."
    echo "  -l          List mode: Lists packages that need to be installed. Does not install."
    echo "  -d          Cleanup mode: Deletes .whl files for packages that are already installed. Does not install."
    echo "  -h          Display this help message."
    echo ""
    echo "Note: The -l and -d options are mutually exclusive."
}

# --- Initialize variables ---
WHL_DIR=""
LIST_ONLY=false
DELETE_ONLY=false

# --- Parse command-line arguments ---
while getopts "p:ldh" opt; do
  case ${opt} in
    p)
      WHL_DIR=$OPTARG
      ;;
    l)
      LIST_ONLY=true
      ;;
    d)
      DELETE_ONLY=true
      ;;
    h)
      usage
      exit 0
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage
      exit 1
      ;;
  esac
done

# --- Validate arguments ---
if [ -z "$WHL_DIR" ]; then
    echo "Error: Path to wheel directory is required. Use -p <path>." >&2
    usage
    exit 1
fi

if $LIST_ONLY && $DELETE_ONLY; then
    echo "Error: Options -l and -d are mutually exclusive." >&2
    usage
    exit 1
fi

if [ ! -d "$WHL_DIR" ]; then
    echo "Error: Directory '$WHL_DIR' not found." >&2
    exit 1
fi

# Use pip to get a list of installed packages.
# The package names are in the first column.
installed_packages=$(pip list --format=freeze | cut -d'=' -f1)

# Check if there are any .whl files in the target directory.
if ! ls "$WHL_DIR"/*.whl &> /dev/null; then
    echo "No .whl files found in '$WHL_DIR'."
    exit 0
fi

# Determine script mode
if $LIST_ONLY; then
    echo "Listing packages to be installed from '$WHL_DIR':"
elif $DELETE_ONLY; then
    echo "Cleaning up already installed packages from '$WHL_DIR':"
else
    echo "Installing packages from '$WHL_DIR':"
fi

# Iterate over all .whl files in the specified directory.
for whl_file in "$WHL_DIR"/*.whl; do
    # Extract package name from the wheel file.
    # This command unzips the METADATA file from the wheel archive in memory
    # and extracts the 'Name:' field. It's robust for various package naming conventions.
    pkg_name=$(unzip -p "$whl_file" "*.dist-info/METADATA" 2>/dev/null | grep -i '^Name:' | cut -d' ' -f2 | tr -d '\r')

    # Check if the package name was successfully extracted.
    if [ -z "$pkg_name" ]; then
        echo "Could not determine package name for $whl_file. Skipping."
        continue
    fi

    # Check if the package is in the list of installed packages.
    # We use grep with word boundaries (-w) and case-insensitivity (-i) for a safe match.
    if echo "$installed_packages" | grep -qiw "^${pkg_name}$"; then
        if $DELETE_ONLY; then
            echo "Package '$pkg_name' is already installed. Deleting '$whl_file'."
            rm "$whl_file"
        elif ! $LIST_ONLY; then
             echo "Package '$pkg_name' is already installed. Skipping installation of '$whl_file'."
        fi
    else
        if $LIST_ONLY; then
            echo "$whl_file"
        elif ! $DELETE_ONLY; then
            echo "Package '$pkg_name' is not installed. Installing '$whl_file'..."
            pip install --no-deps "$whl_file"
        fi
    fi
done

echo "Script finished."