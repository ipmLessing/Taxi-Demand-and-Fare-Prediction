# Description 

This model is for New York taxi fare prediction. We adopt an ensemble learning model which contains a linear regression model, a ridge regression model and one multilayer preception model.

## Structure of the fare prediction

Ensemble learning model - includes a ensemble_learning_model.ipynb on the structure of model.

The ipynb file contains all the code of model testing, data preprocessing, fare model establishment, data testing and metrics evaluation. Detail of the code include these following parts.

### File downloading & dataset loading

This includes the code to download file into Colaboratory (which require Google account login, pls make sure you have a Google account, see in the 'Requirements'), code to import libraries, and code to load the dataset.

### Data preprocessing

This includes the code to define and execute a data preprocessing function, checking the memory usage of dataset, checking null-values in the dataset, and features types.

### To set the test train split

In the dataset, there are 1500000 rows of value in total, the first idx number of rows data will be used for training, while from idx+1 rows the data will be used for testing.
e.g. idx=600000 refers to training set:testing set = 4:6.

### Pre-train tests to ensure correct implementation

This includes:

* Output shape

* Output range

* Variance

* Max error

* RMSE

### Sub-models trainning & fine-tuning

We have 3 sub-models in total, linear regression, ridge regression and MLP regression.

1) Linear regression

* GridSearchCV is used for 2 parameters choosing:

  * 'normalize': (True, False)
  * 'fit_intercept': (True, False)

2) Ridge regression

* GridSearchCV is used for 4 parameters choosing:

  * 'alpha': (1.0, 0.8, 0.6, 0.4, 0.2)
  * 'fit_intercept':(True,False)
  * 'normalize':(True,False)
  * 'max_iter':(500,1000,1500,2000)

3) MLP regression

* GridSearchCV is used for 2 parameters choosing:

  * 'hidden_layer_sizes': (10, 20, 30)
  * 'max_iter': (1, 2, 4, 5)
  
After parameter choosing, best parameter combinations will be printed, then implemented to model fitting.

RMSE will be calculated for each model evaluation.

### Weight choosing for establishing ensemble model

Each sub-model will be implemented a weight to combine the ensemble model. The weights are in betwwen 0 and 1, in a total of 1. Each weight will be tested from 0.05 to 0.95 in an interval of 0.05, then the RMSEs for each combination will be recorded and the best one will be printed.

### Plotting the results

This includes sub-model results plotting and ensemble model result plotting.

### Ensemble Model evaluation to ensure satisfactory performance

Model running time is used to evaluate.

## Running the code

Our code can be run directly on the colab. If you want to run it, you can upload it to the Google Colab and click "Run All" button.

Before starting running the code, pls see the 'Requirements' module.

## Requirements

### Python environment

This is a python environment for running you code, by doing this user will be able to setup exactly the same environment as YOURS. Please drop down your python version.

Go to the python environment location with the terminal and input this command the requirement file will be generated:

```
$ pip freeze > requirements.txt
```

For user who want to install the environment with python, please do followings:
1. install the conda or eb-virt 
2. create the env with the conda or eb-virt

For example with python3.7 and conda:
```
$ conda create -n env python==3.7
$ conda activate env
$ pip install -r requirements.txt
```

*This virtual env will be installed with all the package that specificed in the requirements.txt* in order to run the program

### Register a Google account

The dataset is stored in Google drive. When running the code, Google account should be logged in and get access to the dataset.

To register a Google account, pls go to https://www.google.com/account/about/ and click "create an account".
