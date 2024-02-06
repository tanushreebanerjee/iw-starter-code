## 3. Standard Repository Structure

Establishing a clear and organized repository structure is crucial for maintaining a well-structured and manageable project. Here, we'll outline a basic standard repository structure. Please make sure to add any directories with output from your code as well as directories with datasets and other large files to your gitignore.

### Directory Structure Overview:

1. **root/**:
    - Contains the main project files and directories.
    - Typically named after your project.

2. **data/**:
    - Contains subdirectories for each dataset used in the project.
    - Keeps raw data, processed data, and any related metadata separate from code.
    - Each dataset should have its own subdirectory within the `data` directory.

3. **submodules/**:
    - Holds external dependencies or submodules required by the project.
    - Each submodule may have its own directory structure and version control.
    - Allows for easy management and integration of external code.

4. **code/**:
    - Contains all project-specific code files.
    - Organized into subdirectories based on functionality or module.

5. **experiments/** Store experiment results in this directory. Each experiment should have its own subdirectory containing logs, metrics, visualizations, and any other relevant data.

6. **models/** After training models, save them in this directory. This makes it convenient to load and reuse trained models for inference or further experimentation

7. **.cache/**: Use this directory for caching intermediate results or temporary files generated during data preprocessing, model training, or evaluation. Caching helps improve efficiency by avoiding redundant computations.

### Example Repository Structure:

```
root/
│
├── data/               # Directory for storing datasets
│   ├── dataset1/
│   ├── dataset2/
│   └── ...
│
├── submodules/         # Directory for storing submodule repositories
│   ├── submodule1/
│   ├── submodule2/
│   └── ...
│
├── experiments/         # Directory for experiment results
│   ├── experiment1/
│   ├── experiment2/
│   └── ...
│
├── models/              # Directory for storing trained models
│   ├── model1.pth
│   ├── model2.pth
│   └── ...
│
├── .cache/              # Directory for caching intermediate results or temporary files
│
└── code/                # Directory for project-specific code
    ├── models/          # Stores implementation files for different models used in the project.
    │   ├── model1.py
    │   ├── model2.py
    │   └── ...
    │
    ├── utils/        # Contains utility functions and helper scripts used across the project.
    │   ├── data_utils.py
    │   ├── visualization_utils.py
    │   └── ...
    │
    ├── experiments/
    │   ├── experiment1/
    │   ├── experiment2/
    │   └── ...
    │
    ├── datasets/      # Dataset classes for each dataset, extending PyTorch data class, may also contain code for evaluation 
    │   ├── dataset1/
    │   ├── dataset2/
    │   └── ...
    │
    ├── trainer.py # trainer object with main training loop
    │
    └── main.py # main code that runs full pipeline

```