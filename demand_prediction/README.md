# Description 

This fold contains:

1. two files Model_(best_parameter*).py  correspond to MLP-MDN model & time-series resampling-LSTM model trainning.
2. two h5 files respectively correspond to MLP-MDN model & time-series resampling-LSTM model structure.
3. two Model_pre.py  to load the h5 file and do forecasting.
4. taxi_zones.json is the lookup table of LocationID.
5. time_series_preprocess.py is the preproccesing layer of LSTM model, run this file before run LSTM model.


The model structure is shown in the report figure.18  LSTM-MDN model, LSTM model need pre-preccessed time-series data, run time_series_preprocces.py before do the prediction.


## instructions of HOW TO RUN THE CODE


_data loading_

the data we use in the taining is from June to September， for compelet dataset, pleas download them from https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page.


to run the LSTM Pridiction first prepare the data from (t-10 to t-1), save it as csv file .
```
python time_series_preprocces.py --input "(csv filepath*)"
```
then run
```
import LSTM_model_pre as LSTM
LSTM.model_predict()
```
it will ruturn a dic, predict the demand for next hour

to run the MLP-MDN Pridiction
```
import model_pre as MDN
MDN.model_predict('2018-07-02 07:15:00')
```
it will return a Two-dimensional contour map and a dictionary containing predicted values



## Tuning parameter

to train the MLP-MDN,just run Model_M=40_n1=100_n2=100sigmoid.py in Command Prompt,you can change the parameter at line 334~352 in Model_M=40_n1=100_n2=100sigmoid.py
```
python Model_M=40_n1=100_n2=100sigmoid.py
python LSTM_n=600n=600sigmoid.py
```


to train the LSTM first run time_series_preprocces.py, then run LSTM_M=40_n1=100_n2=100sigmoid.py, you can change the parameter at line 42~50 in Model_M=40_n1=100_n2=100sigmoid.py
```
python time_series_preprocces.py --input "(trainning data filepath*)"
python LSTM_n=600n=600sigmoid.py
```
Requirements.txt 

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


