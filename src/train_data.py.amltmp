
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

parser.add_argument("--prepped_data", type=str , dest = 'prepped_data')
parser.add_argument("--model_output", type=str)
# parser.add_argument("--l2_rate", type=float)

args = parser.parse_args()

prep_data = args.prepped_data

# Load the prep_data file
save_path = os.path.join(prep_data , 'prepped_data.csv')
df = pd.read_csv(save_path)


# Train test split
x,y = df[['Year', 'Month', 'DayofMonth', 'DayOfWeek','OriginAirportID', 'DestAirportID',  'DepDel15','Cancelled','DepDelay']], df['ArrDelay'] 

# removed carrier from x -> 'Carrier'

xtrain,xtest, ytrain, ytest = train_test_split(x,y,train_size=0.8,random_state=42)

# Model Training
model = Ridge(alpha = 0.0).fit(xtrain,ytrain)

# Predict 
ypred = model.predict(xtest)   

# Evaluate
score = model.score(xtest,ytest)
mse = mean_squared_error(ytest,ypred)

# Log metrics
mlflow.log_metric("score", score)
mlflow.log_metric("mean_squared_error", mse)
# mlflow.log_metric("L2_rate", alpha)

# Complete run
mlflow.end_run()\

# save model

output_dir = args.model_output

os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(output_dir, "model.pkl")

joblib.dump(model, model_path)

print("Saved model at:", model_path)


# # save model
# # Current script location
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Parent folder -> PPipeline_Project
# project_dir = os.path.dirname(current_dir)

# # outputs folder
# output_dir = os.path.join(project_dir, "outputs")

# os.makedirs(output_dir, exist_ok=True)

# # Save model
# model_path = os.path.join(output_dir, "model.pkl")

# # model_path = os.path.join(save_path, "model.pkl")



# joblib.dump(model,model_path )
# # joblib.dump(model,save_path )


