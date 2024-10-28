# ML_BigData_Learning

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Overview

This project is part of my studies and professional work in the Business Intelligence department in the Finance domain. The goal is to detect fraudulent credit card transactions using machine learning techniques and graph-based analysis.

The project utilizes the Kaggle Credit Card Fraud Detection dataset and applies a combination of traditional models like Logistic Regression, Decision Trees, and Random Forests, along with graph-based metrics to detect anomalies in transaction behavior.


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         Credit_Card_Fraud_Detection and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── Credit_Card_Fraud_Detection   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes Credit_Card_Fraud_Detection a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

Key Features

    Logistic Regression, Decision Trees, Random Forests: Applied to detect fraudulent transactions.
    Graph-based Analysis: Leverages NetworkX to calculate graph metrics like betweenness centrality to analyze relationships between credit cards and merchants.
    SMOTE (Synthetic Minority Over-sampling Technique): Balances the dataset to handle class imbalance between fraud and non-fraud cases.
    ROC-AUC Score: Measures the model’s ability to distinguish between fraud and non-fraud.

Data

    The dataset used is from Kaggle's Credit Card Fraud Detection. https://www.kaggle.com/code/hamzasafwan/credit-card-fraud-detection/input
    It consists of credit card transactions made by European cardholders in September 2013.
    Features include transaction amount, merchant information, and customer demographics.


Usage

    Prepare the dataset by running the data processing notebook.
    Train and evaluate models by running the respective notebooks (e.g., logistic regression, decision tree, etc.).
    Save and visualize results, including graphs and model performance metrics.

Results

    Logistic Regression Balanced: ROC AUC of 0.9552 on the training set.
    Decision Tree Model: ROC AUC of 0.9765 on the test set.

Next Steps

    Further model tuning and exploration using Random Forest and Gradient Boosting (XGBoost).
    Integration with real-time data streams for live fraud detection.

Acknowledgments

This project was completed as part of the Certificate of Advanced Studies in Machine Intelligence at ZHAW and draws from Kaggle’s Credit Card Fraud dataset.
--------

