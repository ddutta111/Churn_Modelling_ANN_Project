# Churn_Modelling_ANN_Project

## Customer Churn Prediction - European Bank

## Business Problem Description
A European bank has observed an unexpected rise in customer churn, with more clients leaving than anticipated. To understand this issue, the bank conducted a study by selecting a random sample of 10,000 customers across Europe and collected data on various demographic and financial characteristics. Over six months, the bank tracked which customers stayed and which left, recording the outcome as a binary variable. The bank seeks a data scientist's expertise to analyze this dataset and identify at-risk customers.

## Business Goal
The primary goal is to develop a geodemographic segmentation model to predict customer churn. By identifying customers most at risk of leaving, the bank can implement targeted strategies to retain them.

## Dataset and Variables
The dataset includes the following variables:

RowNumber: Sequential identifier of rows (irrelevant for analysis).
CustomerID: Unique ID for each customer.
Surname: Customer's last name (irrelevant for prediction).
CreditScore: Numerical score indicating creditworthiness.
Geography: Country of residence (e.g., France, Germany, Spain).
Gender: Customer's gender (Male/Female).
Age: Customer's age.
Tenure: Number of years with the bank.
Balance: Current account balance.
NumOfProducts: Number of bank products used.
HasCrCard: Indicator of credit card ownership (1 = Yes, 0 = No).
IsActiveMember: Indicator of active membership status (1 = Active, 0 = Inactive).
EstimatedSalary: Estimated annual salary.
Exited: Dependent variable indicating churn (1 = Yes, 0 = No).

## Artificial Neural Network (ANN) Model

## Model Objective
The objective is to build an ANN model to predict customer churn. The model will help the bank identify patterns and correlations in the data to take proactive retention measures.

## Methodology Overview

**1.Data Pre-processing:**

-Import the dataset.
-Standardize and encode data.
-Split data into training and test sets

**2. Building the ANN:**
-Initialize the ANN.
-Add input and hidden layers.
-Add the output layer.

**Training the ANN:**
-Compile the model with an optimizer and loss function.
-Train the model on the training data.

**Making Predictions and Model Evaluation:**
-Predict outcomes for new data.
-Evaluate performance using accuracy metrics.

**Libraries Used**
numpy
pandas
tensorflow (v2.17.0)

This README outlines the project scope and provides details on the dataset and Artificial Neural Network Model (Classification) used for predicting customer churn in the European bank in order to provide detailed insight of the bank's customer churning dataset and recommendations for the client to take actions against the customer churning to dveelop performance.

## Author 

Debolina Dutta

LinkedIn: (https://www.linkedin.com/in/duttadebolina/)
