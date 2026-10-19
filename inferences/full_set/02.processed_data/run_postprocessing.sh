#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_DIR="$(pwd)"

cd -- "$SCRIPT_DIR" || exit
python3 recalculate.py
python3 retrieval.py
python3 stats.py

cd -- "$CURRENT_DIR" || exit
