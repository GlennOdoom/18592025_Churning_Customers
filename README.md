# 18592025_Churning_Customers
Customer Churn Prediction using Random Forest and Keras MLP
Overview
This repository contains code for predicting customer churn using a combination of Random Forest and a Multi-Layer Perceptron (MLP) model implemented with Keras. The code includes data preprocessing, feature selection, exploratory data analysis (EDA), model training, and evaluation.

Files
Notebook.ipynb: Jupyter notebook containing the entire code.
Churn_Mod.pkl: Pickle file containing the trained model.
label_encoded.pkl: Pickle file containing the LabelEncoder used for encoding categorical variables.
Usage
Open and run the Jupyter notebook Notebook.ipynb in your preferred environment (e.g., Google Colab).
Ensure that the necessary libraries are installed by running the installation cell.
Load the dataset, perform data preprocessing, and explore the dataset using EDA.
Train a Random Forest classifier and perform feature selection.
Create customer profiles and visualize relationships between features and churn.
Build and optimize a Keras MLP model using GridSearchCV.
Evaluate the best model on the test set and retrain it with the optimal hyperparameters.
Save the final model and LabelEncoder for future use.
Requirements
Python 3.x
Libraries: pandas, scikit-learn, seaborn, matplotlib, keras, joblib
Acknowledgments
The code makes use of scikit-learn, Keras, and other open-source libraries. Make sure to cite and acknowledge their contributions.

The video to the code: https://drive.google.com/file/d/13qoWHMFscg4Pd7GPeLAsHuIp-sDAq5dE/view?usp=drive_link
