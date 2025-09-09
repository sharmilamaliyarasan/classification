# 🔍 Classification Analysis

## 📌 Project Overview
This project is an End-to-End Machine Learning application that performs classification analysis on the Optical Recognition of Handwritten Digits dataset. It includes data preprocessing, dimensionality reduction, model training, evaluation, and visualization. The notebook covers multiple classification algorithms and provides comprehensive evaluation metrics and visualizations.

## 🚀 Features

Data Preprocessing – Standard scaling and PCA for dimensionality reduction

Multiple Classification Algorithms – Support Vector Machine (SVM), Logistic Regression, K-Nearest Neighbors (KNN), Random Forest

Model Evaluation – Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC Curve

Visualization – PCA-based 2D data visualization, confusion matrix heatmap, ROC curves

End-to-End Workflow – From data loading to model evaluation and visualization

## 🛠️ Tech Stack

Python 🐍

Scikit-learn – Classification models, metrics, and preprocessing

Pandas / NumPy – Data handling

Matplotlib / Seaborn – Visualization

UCI ML Repository – Dataset fetching

## 📂 Dataset

Optical Recognition of Handwritten Digits

Source: UCI Machine Learning Repository (ID: 80)

Features: 64 attributes (8x8 image pixels)

Target: 10 classes (digits 0–9)

## ⚙️ Installation & Setup

Install dependencies:

bash

pip install matplotlib seaborn numpy pandas scikit-learn

Run the Jupyter Notebook:

bash

jupyter notebook classification.ipynb

## 📊 Example Workflow

Load Data – Fetch dataset from UCI repository

Preprocess – Scale features using StandardScaler

Reduce Dimensions – Apply PCA for 2D visualization

Train Models – Fit multiple classifiers

Evaluate – Compute accuracy, precision, recall, F1-score, and plot confusion matrices & ROC curves

Visualize – Plot PCA-reduced data and model performance metrics

## 📈 Evaluation Metrics
Accuracy – Overall correctness

Precision – True positives among predicted positives

Recall – True positives among actual positives

F1-Score – Harmonic mean of precision and recall

## 📸 Example Visualizations

2D PCA scatter plot of digit classes

<img width="671" height="545" alt="image" src="https://github.com/user-attachments/assets/ad1d9cba-866e-48c5-aca5-7fe32c27b708" />

