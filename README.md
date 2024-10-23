# Churn_Modelling_ANN_Project_Python

## Customer Churn Prediction - European Bank

## Business Problem Description
A European bank has observed an unexpected rise in customer churn, with more clients leaving than anticipated. To understand this issue, the bank conducted a study by selecting a random sample of 10,000 customers across Europe and collected data on various demographic and financial characteristics. Over six months, the bank tracked which customers stayed and which left, recording the outcome as a binary variable. The bank seeks a data scientist's expertise to analyze this dataset and identify at-risk customers.

## Business Goal
The primary goal is to develop a geodemographic segmentation model to predict customer churn. By identifying customers most at risk of leaving, the bank can implement targeted strategies to retain them. And, for this analysis, I developed an Artifical Neural Network Model through Deep Learning.

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

> Importing libraries, Dataset and Data Pre-processing
```python
#Importing libraries
import numpy as np
import pandas as pd
import tensorflow as tf

#Import the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)
```
> Standardize and encode data.
```python
#Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)

#One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(), [1])], remainder= 'passthrough')
X = np.array(ct.fit_transform(X))
print(X)
```
> Split dataset and Feature Scaling
```python
#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```
> Building the ANN - Adrring input layer, hidden layers and output layer.
```python
#Initialize the ANN
ann = tf.keras.models.Sequential()
#Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
#Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
#Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
```     
> Training the ANN
```python
#Compile the model with an optimizer and loss function.
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)
```
> Making Predictions and Model Evaluation
```python
# Predict test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#Evaluate performance using Confusion and accuracy metrics
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
```
> Predicting the result of a single observation
- The model is capable of predicting the outcome for a single customer, determining whether they are likely to stay with or leave the bank. This prediction is based on the customer's specific data, using the trained ANN model to assess their churn risk.

For instance, we use our ANN model to predict if the customer with the following informations will leave the bank:

Geography: France

Credit Score: 600

Gender: Male

Age: 40 years old

Tenure: 3 years

Balance: $ 60000

Number of Products: 2

Does this customer have a credit card? Yes

Is this customer an Active Member: Yes

Estimated Salary: $ 50000

So, should we say goodbye to that customer?
```python
#Predicting single customer's churn rate
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))> 0.5)
```
This README outlines the project scope and provides details on the dataset and Artificial Neural Network Model (Classification) used for predicting customer churn in the European bank in order to provide detailed insight of the bank's customer churning dataset and recommendations for the client to take actions against the customer churning to dveelop performance.

## Author 

Debolina Dutta

LinkedIn: (https://www.linkedin.com/in/duttadebolina/)
