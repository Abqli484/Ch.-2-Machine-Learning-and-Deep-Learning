# Chapter 2: Machine Learning and Deep Learning

This repository contains assignments on machine learning and deep learning techniques, focusing on practical applications in classification, clustering, regression, and fraud detection. Each notebook utilizes various algorithms and frameworks to provide hands-on experience with real-world data.

## Notebooks Overview

### 1. Customer Exit Prediction (02_Kelompok_J_1.ipynb)
This notebook develops a classification model to predict customer churn for a bank, using key features from a banking dataset.

- **Dataset:** `SC_HW1_bank_data.csv`
- **Libraries:** Pandas, NumPy, Scikit-learn

#### Key Steps:
- **Data Preprocessing:** Irrelevant columns are removed, followed by one-hot encoding and normalization.
- **Modeling:** Trains and evaluates Logistic Regression, Random Forest, and SVM, .
- **Hyperparameter Tuning:** Uses grid search to find optimal parameters.
- **Evaluation:** Model performance is assessed with accuracy, classification reports, and confusion matrices.

**Result:** The SVM achieved the highest accuracy with efficient processing time.

### 2. Data Segmentation with KMeans Clustering (02_Kelompok_J_2.ipynb)
This notebook uses the KMeans algorithm to cluster customer data, applying unsupervised learning techniques.

- **Dataset:** `cluster_s1.csv`
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

#### Key Steps:
- **Data Preparation:** Removes irrelevant columns and preprocesses the data.
- **Optimal Cluster Selection:** Determines the ideal number of clusters based on Silhouette Score.
- **Clustering and Visualization:** Applies KMeans and visualizes the clustered data.

**Result:** The optimal k-value is chosen based on the highest Silhouette Score, and a scatter plot shows the clustering results.

### 3. California House Price Prediction with Neural Networks (02_Kelompok_J_3.ipynb)
This assignment builds a neural network using TensorFlow and Keras to predict house prices in California.

- **Dataset:** California House Price dataset (from Scikit-Learn)
- **Libraries:** Pandas, NumPy, TensorFlow, Keras, Scikit-learn, Matplotlib

#### Key Steps:
- **Data Preparation:** Splits data into training, validation, and test sets with standardization and normalization.
- **Model Construction:** Creates a Multilayer Perceptron (MLP) with two input layers.
- **Training and Evaluation:** Trains the model, monitoring loss to prevent overfitting.
- **Model Saving:** Saves the trained model for future predictions.

**Result:** The neural network provides accurate house price predictions, with metrics and visualizations indicating strong model performance.

### 4. Fraud Detection in Credit Card Transactions (02_Kelompok_J_4.ipynb)
This notebook builds a PyTorch classification model to detect fraudulent transactions in a credit card dataset.

- **Dataset:** Credit Card Fraud 2023 dataset
- **Libraries:** Pandas, cuDF, cuML, NumPy (cuPy), Scikit-learn, PyTorch

#### Key Steps:
- **GPU Data Loading:** Loads and preprocesses the dataset on GPU for improved performance.
- **Data Conversion:** Converts the data to tensors for PyTorch compatibility.
- **Model Construction:** Builds a Multilayer Perceptron with four hidden layers.
- **Training and Tuning:** Trains the model, targeting an accuracy of at least 95%.

**Result:** The model achieves high accuracy, effectively detecting fraudulent transactions.

## Running the Notebooks

1. Open **Google Colab**.
2. Upload the notebook files (`02_Kelompok_J_1.ipynb` to `02_Kelompok_J_4.ipynb`) to Colab.
3. Follow the instructions in each notebook to execute all cells sequentially for optimal analysis and results.

