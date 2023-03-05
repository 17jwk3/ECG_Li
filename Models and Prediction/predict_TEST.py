"""
Author: Alexander Koch, Ty Cymbalista, Jack Kay
@github: alexkoch14, tycym, 17jwk3
"""

##libaries 
import string
import serial
#import serial.tools.list_ports
import numpy as np
import pandas as pd
import time
#import ecg_plot
#from bluetooth import *
import heartpy as hp
import matplotlib.pyplot as plt
import statistics
from scipy.signal import find_peaks, resample
from scipy import stats
from tensorflow import keras
from xgboost import XGBClassifier
import warnings 
import os
import random

# Global Parameters
num_sec = 3 #recording duration / 2
sample_rt = 360 #sample frequency
os.chdir("C:\\Users\\kocha\\Documents\\Queen's\\Year 4\\ELEC498\\ECG_Li\\Models and Prediction") #Path to project location
warnings.filterwarnings("ignore") #Ignore heartPy warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Ignore TensorFlow warnings


########################
########################
ecg_data = np.genfromtxt('ECG-Test-2.csv', delimiter=',')
ecg_data = resample(ecg_data, num_sec*2*sample_rt)
#######################
#######################

def read_ecg():
        ###########################################
        #        Read ECG from Arduino
        ###########################################

        load_data = []
        baudrate = 230400

        ## COM port setup
        sers = {}
        key = 'COM4'
        sers[key] = serial.Serial(key, baudrate)

        ##COM to Data
        pf = 230400 #polling freqeuncy 

        t_end = time.time() + (num_sec*2)
        while time.time() < t_end:
                #read serial
                values = (sers[key].read(sers[key].inWaiting()).decode())
                #read serial via BT
                #values = serial.Serial(port='COM5', baudrate=baudrate, timeout=0, parity=serial.PARITY_EVEN, stopbits=1)
                #split string into 4 disctint readings
                values = values.strip().split('\r\n')
                for val in values:
                        #append each value to a list
                        load_data.append(val)
                time.sleep(1/360)
        
        ecg_data = load_data
        
        return ecg_data

def predict_stress(ecg_data):
        ###########################################
        #        Stress Detection
        ###########################################

        HRV_data = hp.enhance_ecg_peaks(ecg_data, sample_rt)
        wd, m = hp.process(HRV_data, sample_rt, report_time=False, calc_freq=True)

        # Optional: plot HeartPy enhanced ECG signal
        #plot_object = hp.plotter(wd, m, show=False, title='HeartPy enhaned ECG')
        #plt.show()

        # Calculate HRV metrics
        MEAN_RR = sum(wd['RR_list'])/len(wd['RR_list'])
        MEDIAN_RR = statistics.median(wd['RR_list'])
        SDRR = m['sdnn']
        RMSSD = m['rmssd']
        SDSD = m['sdsd']
        HR = m['bpm']
        pNN50 = m['pnn50']
        SD1 = m['sd1']
        SD2 = m['sd2']

        HRV_metrics = [[MEAN_RR, MEDIAN_RR, SDRR, RMSSD, SDSD,
                        HR, pNN50, SD1,SD2]]

        # Predict stress 
        stress_model = XGBClassifier()
        stress_model.load_model("stress_model_v2.json")
        stress_output = stress_model.predict(HRV_metrics)
        print(stress_output)
        return

def predict_arrhythmia(ecg_data):
        ###########################################
        #        Arrhythimia Detection
        ###########################################

        thresh = 0.5 #Arryhtmia certainty threshold
        test_case = 1000 #Test sample used to validate CNN with train data


        # Filtering code
        ###
        ###


        #Z - Score Data
        ecg_data = stats.zscore(ecg_data)
        ecg_data = ecg_data.reshape(ecg_data.shape[0], 1)


        training_data = np.load('arr_data.npz')
        X_all = training_data['a']
        Y_all = training_data['b']
        symbols = training_data['c']

        X_train = np.reshape(X_all, (X_all.shape[0], X_all.shape[1], 1))

        arr_model = keras.models.load_model('arr_model_v1')

        #arr_output = arr_model.predict(np.array([ecg_data,]))
        #print(arr_output)


        if arr_model.predict(np.array([X_train[test_case],])) > thresh:
                arr_output = 1
        else: arr_output = 0
        print('Model Prediction')
        print(arr_output)
        print('Ground Truth')
        print(Y_all[test_case])
        return

predict_stress(ecg_data)
predict_arrhythmia(ecg_data)