## Federated Time-Series Forecasting
This is the code accompanying the submission to the [Federated Traffic Prediction for 5G and Beyond Challenge](https://supercom.cttc.es/index.php/ai-challenge-2022) of the [Euclid](https://euclid.ee.duth.gr/) team and the corresponding paper entitled "[Federated Learning for 5G Base Station Traffic Forecasting](https://www.sciencedirect.com/science/article/abs/pii/S138912862300395X)" by Vasileios Perifanis, Nikolaos Pavlidis, Remous-Aris Koutsiamanis, Pavlos S. Efraimidis, 2022.

An extension of this work with an in-depth analysis of the energy consumption of the corresponding machine learning models was presented in [2023 Eighth International Conference on Fog and Mobile Edge Computing (FMEC)](https://ieeexplore.ieee.org/xpl/conhome/10305711/proceeding) with the paper entitled "[Towards Energy-Aware Federated Traffic Prediction for Cellular Networks
](https://ieeexplore.ieee.org/abstract/document/10306017)" by Vasileios Perifanis, Nikolaos Pavlidis, Selim F. Yilmaz, Francesc Wilhelmi, Elia Guerra, Marco Miozzo, Pavlos S. Efraimidis, Paolo Dini, Remous-Aris Koutsiamanis.

Finally, this repository includes the extension presented in the Paper [Evaluation of Bio-Inspired Models under Different Learning Settings For Energy Efficiency in Network Traffic Prediction](https://arxiv.org/pdf/2412.17565?) that evaluates emerging bio-inspired models such as Spiking Neural Networks and Reservoir-Computing to address the challenge of Energy Efficiency.

---

This code can serve as benchmark for federated time-series forecasting. 
We focus on raw LTE data and train a global federated model using the measurements of three different base stations on different time intervals. 
Specifically, we implement 6 different model architectures (MLP, RNN, LSTM, GRU, CNN, Dual-Attention LSTM Autoencoder)
and 9 different federated aggregation algorithms (SimpleAvg, MedianAvg, FedAvg, FedProx, FedAvgM, FedNova, FedAdagrad, FedYogi, FedAdam)
on a non-iid setting with distribution, quantity and temporal skew.

### Installation

We recommend using a conda environment with Python 3.8

1. First install [PyTorch](https://pytorch.org/get-started/locally/)
```
$ conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

2. Install additional dependencies
```
$ pip install pandas scikit_learn matplotlib seaborn colorcet scipy h5py carbontracker notebook
```

You can also use the requirements' specification:
```
$ pip install -r requirements.txt
```

### Project Structure
    .
    ├── dataset                 # .csv files
    ├──── ...
    ├── ml                      # Machine learning-specific scipts
    ├──── fl                    # Federated learning utilities
    ├────── client              # Client representation
    ├─────── ...
    ├────── history             # Keeps track of local and global training history
    ├─────── ...
    ├────── server              # Server Implementation
    ├─────── client_manager.py  # Client manager abstract representation and implementation
    ├─────── client_proxy.py    # Client abstract representation on the server side
    ├─────── server.py          # Implements the training logic of federated learning
    ├─────── aggregation        # Implements the aggregation function
    ├───────── ...
    ├─────── defaults.py        # Default methods for client creation and weighted metrics
    ├─────── client_proxy.py    # PyTorch client proxy implementation
    ├─────── torch_client.py    # PyTorch client implementation
    ├──── models                # PyTorch models
    ├───── ...
    ├──── utils                 # Utilities which are common in Centralized and FL settings
    ├────── data_utils.py       # Data pre-processing
    ├────── helpers.py          # Training helper functions
    ├────── train_utils.py      # Training pipeline 
    ├── notebooks               # Tools and utilities
    └── README.md

### Examples
Refer to [notebooks](notebooks) for usage examples.

### Dataset
For an extensive overview of the data collection and processing procedure please refer to [datataset](dataset).
