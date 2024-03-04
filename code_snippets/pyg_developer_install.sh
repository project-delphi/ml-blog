#!/bin/bash
# This script sets up a Python virtual environment and installs PyTorch and PyTorch Geometric.

# Set variables
DIR="$HOME/Code/throwaway/pytorch-geometric-user-install"
RECENT_TORCH_VERSION=2.2.0
YOUR_USERNAME="project-delphi"

# Create the directory and navigate into it
mkdir -p "$DIR"
cd "$DIR"

# Create a Python virtual environment and activate it
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install -q --upgrade pip

# Install dependencies for torch
pip install -q numpy  # to remove user warning with torch install
pip install -q mpmath==1.3.0  # bugfix

# Install Xcode if not already installed
xcode-select --install  > /dev/null 2>&1

# Install Miniconda for Apple Silicon if not already installed
if [ -d "$HOME/anaconda3" ] || [ -d "$HOME/miniconda3" ]
then
    echo "Conda is installed"
else
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
    sh Miniconda3-latest-MacOSX-arm64.sh -b
fi

# Install torch
pip install -q --pre torch==$RECENT_TORCH_VERSION torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# Uninstall torch-geometric if it's already installed
pip uninstall -y torch-geometric

# Clone the PyTorch Geometric repository and install it
git clone "https://github.com/$YOUR_USERNAME/pytorch_geometric"
cd pytorch_geometric
pip install -e ".[dev,full]"

# Check versions
python --version
python -c "import torch; print(f'torch version: {torch.__version__}')"
python -c "import torch_geometric as pyg; print(f'torch geometric version: {pyg.__version__}')"

# Run tests
pytest
