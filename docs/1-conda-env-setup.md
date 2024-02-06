# 1. Conda Environment Setup

Conda is a package manager and environment management system that simplifies the installation and management of software packages and their dependencies. Follow these steps to set up your Conda environment:

### Step 1: Install Conda

If you haven't already installed Conda, you can download and install Miniconda or Anaconda from the official website: [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution). Miniconda is preferred since it is quicker to install and is more lightweight, and is likely suitable for your project.

### Step 2: Create a New Conda Environment

Once Conda is installed, you can create a new environment for your project using the following command:

```bash
conda create --name myproject python=3.8
```

Replace `myproject` with the name you want to give to your environment.

### Step 3: Activate the Environment

Activate the newly created environment with the following command:

```bash
conda activate myproject
```

### Step 4: Install PyTorch and Other Libraries

Now that your environment is activated, you can install PyTorch and other necessary libraries using Conda or pip. 

Note: **To install PyTorch and other large packages, please make sure to use Conda to install them** since Conda has more thorough dependency checks which makes it less likely for the environment to have any dependency issues. To install the latest version of PyTorch, please follow the instructions [here](https://pytorch.org/get-started/locally/). For older versions of PyTorch, refer to [this](https://pytorch.org/get-started/previous-versions/). Installing older versions may be necessary in case you are using another repository as your starter code or as a submodule. For more information on submodules, please see [this doc](docs/2-submodules.md)
