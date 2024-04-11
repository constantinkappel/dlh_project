#!/bin/bash

# Create virtual environment
python -v venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create data directory (check if it exists)
if [ ! -d "/home/$USER/data"]; then
    mkdir /home/$USER/data
fi

# Download pretrained models to checkpoints
#wget ... -o checkpoints/...
