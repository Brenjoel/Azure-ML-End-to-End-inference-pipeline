Environemt.yml
    This file contains all the packages that are required to run this project
exe.ipynb
    This is a python notebook, that has code to connect to an MLclient, create an environment to run the job or script
src/FlightDelay.py
    This file contains the script which does the training and prediction work. This also logs some metrics to compare the performance of the model. Ridge regression is used to train the model and multiple alpha values are passed to the ridge regression as a parameter to observe the changes.

Data is retrived from azure storage account as URI_File