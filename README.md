## Pseudo Slice-Aware Federated Traffic Prediction: Enhancing 5G Traffic Prediction Using Multi-Step Learning and Pseudo-Slicing

To realize our objective, we enhance an existing open-source federated learning (FL) framework for LTE/5G traffic forecasting (Perifanis et al., 2022) with new components aimed at improving accuracy, practicality, and slice-awareness. Below is the planned pipeline of tasks, along with the techniques and justifications:

### 1. Baseline Framework Reproduction (Original Work)
Justification:
Reproducing the original baseline ensures correctness and provides a reference point to measure the impact of our enhancements. It also familiarizes us with the existing architecture and dataset, enabling smoother integration of additional modules.
Technique Used: 
LSTM-based one-step time-series forecasting using the Flower FL framework.
How:
We reproduce the original framework presented by Perifanis et al. (2022), which uses Long Short-Term Memory (LSTM) models for one-step-ahead traffic forecasting. The existing GitHub repository is used to train and evaluate both centralized and federated versions of the model on LTE base station traffic logs. The model input consists of historical traffic features, and the target is the next time step’s traffic volume.

### 2. Multi-Step Forecasting with LSTM (Enhancement)
Justification:
Multi-step forecasting provides advance visibility into traffic trends, which is crucial for real-time network management tasks like proactive scaling or QoS control. This enhancement makes the system more practical and applicable in dynamic telecom environments
Technique Used: 
Sequence-to-Sequence (Seq2Seq) LSTM, Sliding Window, MAE/MSE loss over multiple output steps.
How:
We enhance the LSTM model to forecast multiple future traffic values instead of just the next step. This is achieved by adjusting the output layer to produce a vector of predictions (e.g., t+1 to t+6). We use either a sliding window approach, where past time steps are used to predict future values, or a Seq2Seq model that directly maps input sequences to output sequences. Loss functions such as MAE or MSE are computed across all predicted steps to encourage better temporal modelling.

### 3. Pseudo-Slice Labelling via KMeans Clustering (Enhancement)
Justification:
This approach provides slice-level (service based) differentiation without requiring proprietary slice identifiers from the 5G core. It enables slice-aware training and evaluation, allowing to model traffic behaviour more realistically in a federated learning context.
Technique Used: 
KMeans clustering on traffic features (rnti_count, rb_down, mcs_up, mcs_down, etc.)
How:
Since actual slice information is not available in the dataset, we simulate slice behaviour using unsupervised clustering. KMeans is applied to normalized traffic features such as active user count (rnti_count), throughput (up, down), modulation and coding schemes (mcs_up, mcs_down), and resource block usage. Each resulting cluster is labeled as a pseudo-slice, roughly corresponding to traffic classes like eMBB, URLLC, or mMTC.

### 4. Slice-Aware Client Partitioning in Federated Learning (Enhancement)
Justification:
Including the pseudo-slice label as a feature allows the model to learn behavioural traffic differences while preserving the natural grouping of data per base station. This avoids excessive fragmentation of client data and improves convergence while still enabling slice-aware learning. It enhances model generalization by embedding slice-type awareness directly into the input space, making the approach more scalable and easier to implement in real networks where per-slice data segmentation may not be feasible.
Technique Used: 
Pseudo-slice clustering label added as a feature column; clients grouped per base station; models trained with cluster context; FedAvg for model aggregation.
How:
In this enhancement, we use the KMeans clustering step to assign a pseudo-slice label to each data point and add it as a new feature column in the dataset. This label indicates the traffic behavior pattern (e.g., eMBB-like, URLLC-like, mMTC-like) for that specific timestamp. Federated learning clients are still grouped by base station, but instead of separating data physically per slice, each client's model learns from local data that includes the pseudo-slice context. The cluster label is treated as a categorical or one-hot encoded input to the model, allowing it to distinguish between different traffic patterns while training. Model updates from each client are aggregated using the FedAvg algorithm through the Flower framework.

### 6.	Transformer-Based Forecasting Models (Enhancement)
Justification:
Transformers can outperform recurrent models like LSTM when it comes to learning complex, long-term patterns, especially in time-series data. Evaluating them alongside LSTMs helps benchmark performance and provides architectural insights for 5G traffic modelling.
Technique Used: 
TimesNet or Informer (attention-based time-series models), trained under FL and centralized setups.
How:
We incorporate transformer architectures such as TimesNet or Informer to model long-range temporal dependencies in the traffic data. These models are adapted to perform multi-step forecasting and are trained using both centralized and federated configurations. Inputs are the same as the LSTM models, and output is a sequence of predicted traffic values.

### 6. Real-Time Dashboard and Visualization (Enhancement)
Justification:
Interactive visualization makes the forecasting outcomes understandable for both technical and non-technical users. It adds value to the project presentation and is particularly useful in academic evaluations and live demos.
Technique Used: 
Streamlit web framework, historical data replay for simulation.
How:
To demonstrate model behaviour interactively, we build a dashboard using Streamlit. It replays historical logs row-by-row, simulating real-time input. The interface displays predicted vs. actual traffic per pseudo-slice and allows users to switch between models and base stations. Visual elements include time-series plots, prediction intervals, and performance metrics.



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
    ├── capstone                # Files for enactments on base paper
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
