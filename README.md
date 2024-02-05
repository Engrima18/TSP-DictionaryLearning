# TSP-DictionaryLearning

## Table of Contents

- [TSP-DictionaryLearning](#project-name)
  - [Table of Contents](#table-of-contents)
  - [Project Description](#project-description)
  - [Installation](#installation)
    - [Clone the Repository](#clone-the-repository)
    - [Setup Environment](#setup-environment)
    - [Install Dependencies](#install-dependencies)
  - [Usage](#usage)
    - [Running the Simulation](#running-the-simulation)
    - [Customizing Parameters](#customizing-parameters)
  - [Project Structure](#project-structure)
  - [Running Tests](#running-tests)

## Project Description

## Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/yourrepositoryname.git
cd yourrepositoryname
```

### Setup Environment
It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Install Dependencies
Install all required dependencies:

```bash
pip install -r requirements.txt
```
## Usage

### Running the Simulation
Navigate to the project directory and run:

```bash
python scripts/main.py
```

### Customizing Parameters
You can customize the simulation with various command-line arguments:

```bash
python scripts/main.py --n_sim 20 --n 40 --p 0.2
```
For more details on available parameters:

```bash
python scripts/main.py --help
```

## Project Structure

```bash
TSP-DictionaryLearning/
│
├── tsplearn/       # Package directory
│   ├── __init__.py        # Initializes the package
│   ├── curves_plot.py     # Module file
│   ├── data_gen.py        # Topological signal and dictionary generation
│   ├── model_train.py     # Alternated optimization algorithm for dictionary learning
│   └── tsp_utils.py       # Auxiliary class for graph definition
|   
│
├── samples/        # Examples and sample usage
│   ├── main.py            # Example of complete simulation run
│   └── res.py              # Example auxiliary script for showing and saving results
│
├── tests/                 # Unit tests
│   └── ...
│
├── README.md              # Project overview and usage instructions
├── requirements.txt       # Dependencies list
└── setup.py               # Package installation script
```