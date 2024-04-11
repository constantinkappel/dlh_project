#!/bin/bash

# Create virtual environment
python -v venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create data directory
mkdir /home/$USER/data

# Download pretrained models to checkpoints
#wget ... -o checkpoints/...
