# 🛡️ FinGraph AML System

**AI-Powered Anti-Money Laundering Detection using Graph Neural Networks**

> **Note for Recruiters/Reviewers:** This project is a demonstration of full-stack ML engineering capabilities, ranging from data engineering (DuckDB) to deep learning (PyTorch Geometric) and frontend visualization (Streamlit). It is currently in active development.

## 📖 Project Overview

FinGraph is a system designed to detect complex money laundering typologies (such as Smurfing and Cyclic Laundering) in financial transaction networks. Unlike traditional rule-based systems, FinGraph utilizes **Graph Neural Networks (GATv2)** to learn structural patterns in transaction graphs.

### 📊 Performance (Latest Run)
*   **Recall**: **81.7%** (Successfully identifies 4 out of 5 money launderers)
*   **Accuracy**: ~60% (Prioritizes safety over precision - "Better safe than sorry")
*   **Training Strategy**: Uses **Dynamic Loss Weighting** (1:80 ratio) and **Multi-Task Learning** (Topology + Classification) to handle extreme class imbalance.

## 🛠️ Tech Stack
*   **Language**: Python 3.10
*   **Package Manager**: `uv` (Modern, fast Python package installer)
*   **Data Engine**: **DuckDB** (In-process OLAP database for high-performance CSV querying)
*   **Machine Learning**: **PyTorch Geometric** (Multi-Task GATv2)
    *   **Node Classification**: Detects illicit entities.
    *   **Link Prediction**: Auxiliary task to force topology learning.
*   **Visualization**: **Streamlit** & **Plotly** (Interactive Dashboard)
*   **Infrastructure**: Designed for AWS Lambda (Serverless Inference)

### 🧠 How It Works
Want to understand the math and logic behind the GNN?
👉 **[Read the Algorithm Explanation](docs/ALGORITHM_EXPLAINED.md)**

---

## 🚧 Current Status & Development Log

**Date:** December 9, 2025

### ✅ Completed Modules
1.  **Data Pipeline (`src/fingraph/data`)**:
    *   Automated ingestion of the **IBM Transactions for AML** dataset (Kaggle).
    *   **DuckDB** integration to handle large CSVs (500k+ nodes, 5M+ edges) efficiently in memory.
    *   SQL-based feature extraction (In/Out Degree, Transaction Volumes).
2.  **Training Pipeline (`src/fingraph/core/train.py`)**:
    *   End-to-end training loop using PyTorch Geometric.
    *   Implementation of **GATv2** (Graph Attention Network) with multi-head attention.
    *   Handling of extreme class imbalance (1:80) using `BCEWithLogitsLoss` and class weights.
3.  **Inference Engine (`src/fingraph/core/inference.py`)**:
    *   Standalone module to load trained models and predict risk scores for specific accounts.
    *   "Top K" logic to identify the highest-risk actors in the network.
4.  **Dashboard (`src/fingraph/api/app.py`)**:
    *   Interactive UI to visualize the transaction graph.
    *   Real-time integration with the Inference Engine to show risk scores.

### ⚠️ Current Limitations (Work in Progress)
*   **Binary Classification Focus**: The current model treats the problem as a binary node classification task ("Is this account laundering?"). It does not yet explicitly classify specific typologies (e.g., "This is a Smurfing ring").
*   **Model Convergence**: The model currently faces challenges with **Class Imbalance**. Despite using weighted loss functions, the model tends to be conservative (predicting low probabilities for everyone).
*   **Feature Engineering**: Initial training runs relied heavily on graph structure. We have just added explicit node features (Transaction Volume, Degree Centrality) to help the model distinguish between actors, but these need further tuning.

---

## 🗺️ Roadmap & Future Improvements

To make this a production-grade system, the following steps are planned:

1.  **Refine Training Strategy**:
    *   Implement **Focal Loss** to better handle the "hard" examples in the imbalanced dataset.
    *   Experiment with **SMOTE** or undersampling to balance the training set.
2.  **Advanced Feature Engineering**:
    *   **Temporal Features**: Use RNNs/LSTMs to capture the *timing* of transactions (e.g., "bursts" of activity typical in smurfing).
    *   **Motif Counting**: Explicitly count triangles and 4-cycles as input features.
3.  **Typology Classification**:
    *   Move from Binary Classification to **Multi-class Classification** to identify specific laundering methods.
4.  **Deployment**:
    *   Finalize the **AWS Lambda** container for serverless inference.
    *   Deploy the Streamlit app via Docker.

---

## 🚀 Getting Started

### 1. Installation
This project uses `uv` for fast dependency management.

```bash
# Install uv if needed
pip install uv

# Sync dependencies
uv sync
```

### 2. Training the Model
The training script downloads the data (if missing), builds the graph, and trains the GNN.

```bash
uv run python src/fingraph/core/train.py
```

### 3. Running Inference
Check the model's performance on the top risk accounts.

```bash
uv run python src/fingraph/core/inference.py
```

### 4. Launching the Dashboard
Explore the graph visually.

```bash
uv run streamlit run src/fingraph/api/app.py
```

---

## 📂 Project Structure

```text
src/fingraph/
├── api/
│   └── app.py           # Streamlit Dashboard
├── core/
│   ├── model.py         # GATv2 PyTorch Model Definition
│   ├── train.py         # Training Loop & Optimization
│   └── inference.py     # Prediction Logic
├── data/
│   ├── data_engine.py   # DuckDB Logic & Graph Construction
│   └── download.py      # Kaggle Dataset Downloader
└── utils/
    └── logger.py        # Logging Configuration
```
