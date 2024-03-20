# set variables here
DIR="$HOME/Code/throwaway/pytorch-geometric-developer-install"
PYTHON_VERSION=3.11
RECENT_TORCH_VERSION=2.2.0
GITHUB_USERNAME="project-delphi"
MIN_MACOSX_DEPLOYMENT_TARGET=$(sw_vers -productVersion)

# install miniconda for apple silicon, if not already installed
if [ ! -d "$HOME/anaconda3" ] && [ ! -d "$HOME/miniconda3" ]
then
    echo "installing conda..."
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
    sh Miniconda3-latest-MacOSX-arm64.sh -b -u  > /dev/null 2>&1
fi

mkdir -p "$DIR"
cd "$DIR"
conda create --yes  -p $DIR/.venv python=$PYTHON_VERSION  > /dev/null 2>&1
eval "$(conda shell.bash hook)"
conda activate $DIR/.venv

pip install -q --upgrade pip

##### TORCH BUILD AND INSTALL ON M1, to use GPUs #####
pip install -q numpy # to remove user warning with torch install
pip install -q mpmath==1.3.0 # bugfix
xcode-select --install  > /dev/null 2>&1 # if xcode not installed

###### install pytorch ######
pip install -q --pre torch==$RECENT_TORCH_VERSION torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# install dev build dependencies
pip install -q cmake
pip install -q ninja wheel
pip install -q git+https://github.com/pyg-team/pyg-lib.git
MACOSX_DEPLOYMENT_TARGET=$MIN_MACOSX_DEPLOYMENT_TARGET CC=clang CXX=clang++ python -m pip -q --no-cache-dir  install  torch-scatter
MACOSX_DEPLOYMENT_TARGET=$MIN_MACOSX_DEPLOYMENT_TARGET CC=clang CXX=clang++ python -m pip -q --no-cache-dir  install  torch-sparse
MACOSX_DEPLOYMENT_TARGET=$MIN_MACOSX_DEPLOYMENT_TARGET CC=clang CXX=clang++ python -m pip -q --no-cache-dir  install  torch-cluster
MACOSX_DEPLOYMENT_TARGET=$MIN_MACOSX_DEPLOYMENT_TARGET CC=clang CXX=clang++ python -m pip -q --no-cache-dir  install  torch-spline-conv

# clone the forked repository and rebase to original
git clone "https://github.com/$GITHUB_USERNAME/pytorch_geometric.git"  2>/dev/null
cd pytorch_geometric
if ! git remote | grep -q 'upstream'; then
    git remote add upstream "https://github.com/pyg-team/pytorch_geometric"
fi
git fetch upstream  -q
git rebase upstream/master

# build dev install
MACOSX_DEPLOYMENT_TARGET=$MIN_MACOSX_DEPLOYMENT_TARGET CC=clang CXX=clang++ python -m pip install -q --no-cache-dir -e ".[dev,full]"  #> /dev/null 2>&1

# check
python --version
python -c "import torch; print(f'torch version: {torch.__version__}')"
python -c "import torch_geometric as pyg; print(f'torch geometric version: {pyg.__version__}')"
