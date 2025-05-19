# Graph Neural Networks: Social Spread of Misinformation

Project for the course "Complex Systems" at the University of Coimbra, 2025.

## Authors

- Tomé Filipe Gilberto Ferreira, 2021232615
- Miguel Gameiro Leopoldo, 2021225940

## Brief Description

Online social networks are prime examples of complex systems, where information (credible or not) can spread rapidly through user interactions. In this project, you will explore the emergence of misinformation cascades by leveraging Graph Neural Networks (GNNs) on real-world data. Various publicly available Twitter or Facebook datasets (e.g., FakeNewsNet, available onGitHub) include user–user interactions and post-level content that can be transformed into graph structures. The goal is to analyze how misleading posts propagate, identify influential nodes, and examine conditions underwhich misinformation dominates or fizzles out.

## Goals

From a publicly available social media dataset, represent user interactions (e.g., follows, retweets, replies) as graph edges and embed relevant user or content features as node attributes. The goal is to build a GNN to classify each news item as fake or real, and compare its performance against a baseline. Explore which features (e.g., user credibility, network centrality) the GNN relies on most for detection, and examine whether certain network structures or activity patterns increase the likelihood of fake news proliferation. Finally, discuss any threshold or emergent effects that manifest as misinformation spreads at scale.

## Usage Guide

### 1. Install Dependencies

Make sure you have Python 3.8+ and install the required packages:

```sh
pip install -r requirements.txt
```

### 2. Prepare Data

The dataset will be automatically downloaded to the data folder on first run. No manual download is needed. Just ensure you have an internet connection the first time you run the code.

### 3. Run Experiments

- **Train and sweep models:**  
  Run hyperparameter sweeps and training with:

  ```sh
  python main.py
  ```

- **Test model:**  

Change the parameters in the file according to your needs and run the code.

  ```sh
  python test_model.py
  ```

### 4. Weights & Biases

This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking.  
Set up your account and login with:

```sh
wandb login
```

### 5. Customization

- Change dataset, features, or model parameters by editing the corresponding Python files or the `sweep.yaml` configuration.
