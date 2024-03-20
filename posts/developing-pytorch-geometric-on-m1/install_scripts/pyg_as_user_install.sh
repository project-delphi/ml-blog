# set variables here
DIR="$HOME/Code/throwaway/pytorch-geometric-user-install"
PYTHON_VERSION=3.11
RECENT_TORCH_VERSION=2.2.0

# install miniconda for apple silicon, if not already installed
if [ -d "$HOME/anaconda3" ] || [ -d "$HOME/miniconda3" ]
then
    echo "Conda is installed"
else
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
    sh Miniconda3-latest-MacOSX-arm64.sh -b -u  > /dev/null 2>&1
fi

mkdir -p "$DIR"
cd "$DIR"
conda create --yes  -p $DIR/.venv python=$PYTHON_VERSION > /dev/null 2>&1
eval "$(conda shell.bash hook)"
conda activate $DIR/.venv
pip install -q --upgrade pip

##### TORCH BUILD AND INSTALL ON M1, to use GPUs #####
pip install -q numpy # to remove user warning with torch install
pip install -q mpmath==1.3.0 # bugfix
xcode-select --install  > /dev/null 2>&1 # if xcode not installed

###### install torch ######
pip install -q --pre torch==$RECENT_TORCH_VERSION torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# install torch geometric
pip install -q torch-geometric

# check
python --version
python -c "import torch; print(f'torch version: {torch.__version__}')"
python -c "import torch_geometric as pyg; print(f'torch geometric version: {pyg.__version__}')"
