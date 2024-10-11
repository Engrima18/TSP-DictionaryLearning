# TSP-DictionaryLearning

This is the official repository containing all the code to run the simulations and test the learning algorithm proposed in the master's thesis in Data Science **Joint Topology and Dictionary Learning for Sparse
Data Representation over Cell Complexes** (a.y. 2023/24)

## Table of Contents

- [Abstract](#abstract)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Clone the Repository](#clone-the-repository)
  - [Setup Environment](#setup-environment)

## Abstract

This thesis is aimed to the development of an efficient algorithm for the joint learning of dictionary atoms, sparse representations, and the underlying topology of signals defined over cell complexes. A comprehensive review of Topological Signal Processing is presented, emphasizing signals defined on specific topological domains such as the cell complexes. The study then examines the field of Compressive Sensing and highlights state-of-the-art Dictionary Learning techniques.

Following the theoretical foundation, the research addresses the problem of dictionary learning within a topological context, aiming to compress topological signals by constructing an overcomplete geometry-informed dictionary. This dictionary is created through the concatenation of several sub-dictionaries, where each sub-dictionary functions as a convolutional cell filter (polynomials of the Hodge Laplacians associated with the underlying topological domain), and is parameterized by distinct dictionary coefficients for upper and lower adjacencies, in accordance with a so-called Separated Hodge Laplacian parameterization.

A non-convex optimization problem is formulated for dictionary and sparse representation learning and is efficiently solved through an iterative alternating direction algorithm. The same algorithm is employed to compare the signal approximation results from the proposed parameterization, with those obtained from other topology-based parameterization techniques, as well as from analytical dictionaries based on topological Fourier transform, wavelet transform, and Slepians. The algorithm is further enhanced by incorporating a topology learning step, enabled through a greedy search for the optimal upper Laplacian. This novel approach merges the representative power of overcomplete, learnable dictionaries with the efficiency of controllable optimization methods.

This framework adopts an innovative perspective in signal processing, emphasizing the utilization of geometric structures and algebraic operators for the development of algorithms with robust generalization capabilities. Additionally, it allows for future enhancements, including the incorporation of alternative topology learning techniques and the integration of model-based deep learning methods to further augment the overall algorithm.

## Project Description

The project is fully developed in Python, and the complete code is available in the corresponding GitHub repository. The repository contains two main directories: topolearn, which includes the module with essential functions and classes for implementing the proposed algorithm, and experiments, where scripts used to generate the numerical results on both synthetic and real data, as reported in the thesis, can be found. Both the real-world and synthetically generated datasets are also available for download in their respective folders within the repository.

The core class of the project is TopoSolver, located in the topolearn module. This class manages both the joint topology and dictionary learning procedures introduced in this work, as well as sparse representation using analytical dictionaries, such as Slepians and Hodgelets. For optimization, the project utilizes the MOSEK solver for semidefinite programming (SDP) problems and GUROBI for quadratic reformulations. Both solvers are accessed via the CVXPY library's API to streamline the alternating direction algorithm process.

## Project Structure

  ```bash
  TSP-DictionaryLearning/
  │
  ├── cache/          # Caching and memoizing intermediate results
  |
  ├── config/
  │   ├── algorithm.yaml       # Configuring the algorithm hyperparameters and the used methods
  │   ├── config.yaml          # Main config file for setting-up the topology, the parameters for the data generation process
  │   ├── visualization.yaml   # Config for the aesthetic characteristics of plots
  |
  ├── experiments/             # Folder containing all the experiments and simulations reported in the thesis
  │   ├── analyt_dict_learn.py # Sparse representation with analytical dictionaries
  |   ├── dict_learn.py        # Dictionary learning with parametric dictionaries
  |   ├── dict_topo_learn.py   # Joint topology and dictionary learning
  |   ├── res.py               # Script for obtaining the final results
  |   ├── utils.py
  |
  ├── logs/                    # Directory for the logging files coming from the experiments
  |
  ├── plots/                   # Folder for the automatic save of plots in .png format
  |
  ├── synthetic_data/          # Synthetic dataset for several generating setups
  │   ├── ...
  |
  ├── real_data/  
  │   ├── real_data.mat        # DFN dataset for experiments on real topological signals      
  |
  ├── topolearn/               # Package directory
  │   ├── __init__.py          # Initializes the package
  │   ├── data_generation.py   # Functions for generation of synthetic datasets of topological signals
  │   ├── EnancedGraph.py      # Class for the 2nd-order Cell Complex
  │   ├── Hodgelets.py         # Classes for the implementation of Hodgelets and Slepians
  |   ├── TopoSolver.py        # Main class containing the procedure for joint topology and dictionary learning
  |   ├── utils.py             # Utils functions for memoization and saving plots
  |   ├── utilsHodgelets.py    # Auxiliary functions for Hodgelets and Slepians
  |   ├── utilsTopoSolver.py   # Auxiliary functions for TopoSolver class
  │   └── visualization.py     # Module for plots and results visualization
  │
  |
  ├── README.md              # Project overview and usage instructions
  ├── requirements.txt       # Dependencies list
  └── setup.py               # Package installation script
  ```

## Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/yourrepositoryname.git
cd yourrepositoryname
```

### Setup Environment

## Used technologies

Il progetto è interamente scritto in Python. Tutto è il codice è reperibile alla corrispondente repository github, dove è possibile trovare due folder principali : topolearn contentente il modulo con le funzioni e le classi fondamentali per l'implementazione dell'algoritmo proposto, e experiments dove è sono contenuti gli script in grado di fornire i risultati numerici su dati sintetici e dati reali riportati nella tesi.
è possibile scaricare sia il dataset reale che quelli generati sinteticamente nelle corrispettive cartelle della repository.

La classe principale è TopoSolver, contenuta nel modulo topolearn. Tale classe gestisce sia le procdure di dictionary learning e joint topology e dictionary learning introdotte in questo lavoro, sia la sparse representation con dizionari analitici quali Slepians e Hodgelets. Per l'ottimizzazione tramite algoritmo a direzioni alternate usiamo il solver MOSEK nel caso SDP e GUROBI per la riformulazione in problema quadratico, entrambi i solver sono gestiti tramite API della libreria CVXPY
