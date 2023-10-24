# Invariant Structure Learning for Better Generalization and Causal Explainability

This is an implementation of the paper 
[Invariant Structure Learning for Better Generalization and Causal Explainability](https://openreview.net/pdf?id=A9yn7KTwsK)



- **Synthetic Data Generation**: `data_generation_syn`
- **Real Data Generation**: `data_generation_real`
- **Output**: Different environments.

## ISL Supervised Learning

- **Model Function**: Provides non-linear definitions and operations.
- **Utilities**:
  - DAG operations (save, draw).
  - Weight ranking (`rank w`).
  - Accuracy computations.
  - K-means clustering and more.
  
- **Main Workflow**: `main.py` reads multi-environment data, invokes the model, and utilities, resulting in DAG and Y-related DAG.
- **Strategies**:
  1. Keep all data.
  2. Other strategies (further details needed).
  
- Threshold is available as a function in utilities.
- **Y Predictor**: Implemented using Multi-Layer Perceptron (MLP).

## ISL Unsupervised Learning

- **Aggregation**: Performed manually.
- **Retraining**: Utilizes NOTEAR-MLP.
- **Backup and Initialization**: Load `/theta^{Y}_{1}` and initialize or fix as needed.

## Files and Usage

- `nonlinear.py`: Original NOTEAR-MLP.
  
- `sachs.csv`: Preprocessed Sachs data. Header removed for ease of use with `numpy.loadtxt`.

- `utils.py`: Contains functions for data generation, plotting, clustering across different environments.

- `nonlinear_ourDataNonlinear_daring.py`: Tests NOTEAR on data generated in DARING's nonlinear manner.

- `notearMLP_experiment_environment.py`: Tests across different environments.

- `nonlinear_castle`: Modified NOTEAR-MLP with added Y prediction loss for real data. `y_index` in notear-mlp indicates which feature is treated as Y.

- `sachs_data_NOTEAR-MLP.py`: Tests NOTEAR and its modifications on the real dataset (Sachs protein data).
  ```bash
  python sachs_data_NOTEAR−MLP.py sachs.csv


* To switch to a different version of NOTEAR, modify:
```import ... as nonlinear```

* To use a different feature as label y, change the y_index in notears_nonlinear.
* sachs_cluster: Clusters the Sachs data as different environments and applies NOTEAR-MLP on them.
```python sachs_cluster.py sachs.csv```



