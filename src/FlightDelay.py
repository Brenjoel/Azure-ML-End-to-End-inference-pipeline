
# import statements
# from azureml.core import Workspace , Datastore, Dataset, Run , Experiment



import pandas as pd
import numpy as np
import argparse

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix , accuracy_score, mean_squared_error

import mlflow
import mltable

import os
import joblib

# Parse AML input argument
parser = argparse.ArgumentParser()

parser.add_argument("--input_data", type=str)
parser.add_argument("--l2_rate", type=float)

args = parser.parse_args()

print("Input path:", args.input_data)
l2 = args.l2_rate
alpha = l2

# Load data
df = pd.read_csv(args.input_data)

df["DepDelay"] = df["DepDelay"].fillna(0)
df["DepDel15"] = df["DepDel15"].fillna(0)
df['ArrDelay'] = df["ArrDelay"].fillna(0)


le = LabelEncoder()
df['Carrier'] = le.fit_transform(df['Carrier'])

# Train test split
x,y = df[['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'Carrier','OriginAirportID', 'DestAirportID',  'DepDel15','Cancelled','DepDelay']], df['ArrDelay']



xtrain,xtest, ytrain, ytest = train_test_split(x,y,train_size=0.8,random_state=42)

# Model Training
model = Ridge(alpha = alpha).fit(xtrain,ytrain)

# Predict 
ypred = model.predict(xtest)   

# Evaluate
score = model.score(xtest,ytest)
mse = mean_squared_error(ytest,ypred)

# Log metrics
mlflow.log_metric("score", score)
mlflow.log_metric("mean_squared_error", mse)
mlflow.log_metric("L2_rate", alpha)

# Complete run
mlflow.end_run()

# save model
os.makedirs('Outputs',exist_ok = True)
joblib.dump(model, filename='Outputs/LR_FlightDelayModel.pkl')




