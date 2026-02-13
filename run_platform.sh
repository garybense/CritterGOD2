#!/bin/bash
# Run CritterGOD Research Platform with Python 3.13
# (Python 3.14 has pygame font compatibility issues)

cd "$(dirname "$0")"
PYTHONPATH="$(pwd)" python3.13 examples/research_platform.py "$@"
