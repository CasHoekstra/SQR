# Symbolic Quantile Regression for Interpretable Quantile Predictions
<hr style="border:2px solid gray">

Symbolic Quantile Regression (SQR) for Interpretable Quantile Predictions is an adjusted framework of PySR developed by  <a href="https://github.com/MilesCranmer/PySR"> Cranmer </a> This framework is specifically tailored to do conditional quantile predictions on the SRBench dataset. 

The framework utilizes Genetic Programming techniques to produce explainable mathematical equations that can be utilized to do quantile predictions in an interpretable manner, to be able to investigate target behavior at several different conditional quantiles. 


# Installation Instructions for this Project

## Requirements
- Python 3.10.3 (recommended)
- Julia 1.9+ (https://julialang.org/downloads/)

## Installation

### Using anaconda (recommended)
1. Install the Python package and environment manager [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install).

2. Install from the environment file
    ```bash
    conda env create -f conda_environment.yml
    ```

### Using pip

1. **Create a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Julia and configure Julia dependencies**:
   Open a Julia terminal and execute:
   ```julia
   import Pkg
   Pkg.add(["SymbolicRegression"])
   ```


## Troubleshooting
If `pysr` causes issues, ensure that Julia is correctly installed and update the Julia packages:
```julia
import Pkg
Pkg.update()
```
### Full requirements.txt, also available in repo ###

alembic==1.14.1
appnope==0.1.4
asttokens==3.0.0
certifi==2025.1.31
charset-normalizer==3.4.1
click==8.1.8
colorlog==6.9.0
comm==0.2.2
debugpy==1.8.12
decorator==5.1.1
executing==2.2.0
filelock==3.17.0
fsspec==2025.2.0
idna==3.10
ipykernel==6.29.5
ipython==8.32.0
jedi==0.19.2
Jinja2==3.1.5
joblib==1.4.2
juliacall==0.9.23
juliapkg==0.1.15
jupyter_client==8.6.3
jupyter_core==5.7.2
lightgbm==4.6.0
Mako==1.3.9
MarkupSafe==3.0.2
matplotlib-inline==0.1.7
mpmath==1.3.0
nest-asyncio==1.6.0
networkx==3.4.2
numpy==2.2.3
optuna==4.2.1
packaging==24.2
pandas==2.2.3
parso==0.8.4
patsy==1.0.1
pexpect==4.9.0
platformdirs==4.3.6
pmlb==1.0.1.post3
prompt_toolkit==3.0.50
psutil==7.0.0
ptyprocess==0.7.0
pure_eval==0.2.3
Pygments==2.19.1
pysr==1.4.0
python-dateutil==2.9.0.post0
pytz==2025.1
PyYAML==6.0.2
pyzmq==26.2.1
requests==2.32.3
scikit-learn==1.6.1
scipy==1.15.2
semver==3.0.4
setuptools==75.8.0
six==1.17.0
SQLAlchemy==2.0.38
stack-data==0.6.3
statsmodels==0.14.4
sympy==1.13.1
threadpoolctl==3.5.0
torch==2.6.0
tornado==6.4.2
tqdm==4.67.1
traitlets==5.14.3
typing_extensions==4.12.2
tzdata==2025.1
urllib3==2.3.0
wcwidth==0.2.13
