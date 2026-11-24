#!/usr/bin/env bash
# Automatically navigate to project directory
cd "$(dirname "$0")"

# Activate TensorFlow virtual environment
source /home/omarealmaz/tf-env/bin/activate

echo "=========================================================="
echo "  Starting MNIST CNN Studio & Backend Server"
echo "=========================================================="

# Launch web/app.py
exec python3 web/app.py
