# Fraud Detection Machine Learning Pipeline

## Project Overview
This repository contains a comprehensive machine learning workflow designed for fraud detection. The project implements a dual-phase approach, utilizing unsupervised clustering algorithms to uncover hidden patterns and segment the data, followed by supervised classification models to predict fraudulent activities based on the engineered features and cluster assignments.

## Repository Structure
The project is organized into the following key components:

- **Clustering.ipynb**: The first phase of the pipeline. This notebook handles data preprocessing, exploratory data analysis, and the implementation of various clustering algorithms (e.g., K-Means, DBSCAN, Hierarchical Clustering). It aims to segment the dataset and assign cluster labels that are later used for classification.
- **Classification.ipynb**: The second phase of the pipeline. This notebook takes the processed data and cluster labels from the first phase to train, evaluate, and optimize classification models. It includes techniques to handle data leakage, cross-validation, and comprehensive performance metrics.
- **models/**: Directory intended for saving trained model artifacts (e.g., `.pkl` or `.joblib` files) for future deployment or inference.
- **images/**: Directory for storing generated plots, charts, and visual assets used in the notebooks or documentation.

## Features and Methodology

### 1. Data Preprocessing and Clustering
- Comprehensive data cleaning, handling missing values, and outlier removal.
- Feature scaling and transformation.
- Implementation and comparison of multiple clustering algorithms.
- Evaluation using internal metrics such as Silhouette Score and visual interpretation via PCA (Principal Component Analysis).

### 2. Classification and Predictive Modeling
- Mitigation of data leakage through strategic feature selection and label noise injection.
- Training robust classification models to identify target events (fraud).
- Model evaluation utilizing Precision-Recall curves, learning curves, and standard classification metrics (Accuracy, Precision, Recall, F1-Score).

## Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd "Clustering-Classication ML"
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install required dependencies:**
   Ensure you have the necessary libraries installed. Common requirements include `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `seaborn`.
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

## Usage
To execute the pipeline, run the notebooks sequentially:
1. Open and run `Clustering.ipynb` to process the raw data and generate the clustered dataset.
2. Open and run `Classification.ipynb` to train the predictive models and evaluate the final classification performance.

## License
This project is provided for educational and demonstrative purposes. Please refer to the repository owner for specific licensing details.
