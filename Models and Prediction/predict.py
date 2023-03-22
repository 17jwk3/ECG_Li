"""
Author: Alexander Koch, Ty Cymbalista, Jack Kay
@github: alexkoch14, tycym, 17jwk3
"""

##libaries 
import serial
#import serial.tools.list_ports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import heartpy as hp
import statistics
import scipy.signal as sps
from scipy import stats
import tkinter
import time
from tensorflow import keras
from sklearn.impute import KNNImputer
from hrvanalysis import get_time_domain_features, get_poincare_plot_features
import joblib
import warnings 
import os
import csv

# Global Parameters
num_sec = 3 #recording duration / 2
sample_rt = 360 #sample frequency
os.chdir("C:\\Users\\kocha\\Documents\\Queen's\\Year 4\\ELEC498\\ECG_Li\\Models and Prediction") #Path to project location
warnings.filterwarnings("ignore") #Ignore heartPy warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Ignore TensorFlow warnings



def read_ecg():
        #########Read ECG from Arduino#############
        load_data = []
        baudrate = 115200

        ## COM port setup
        sers = {}
        key = 'COM3'
        sers[key] = serial.Serial(key, baudrate)

        t_end = time.time() + (num_sec*2)
        while time.time() < t_end:
                #read serial
                values = (sers[key].read(sers[key].inWaiting()).decode())
                #read serial via BT
                #values = serial.Serial(port='COM5', baudrate=baudrate, timeout=0, parity=serial.PARITY_EVEN, stopbits=1)
                #split string into 4 disctint readings
                values = values.strip().split('\r\n')
                for val in values:
                        if not val:
                                val = 5
                        val = int(val)
                        #append each value to a list
                        load_data.append(val)
                time.sleep(1/sample_rt)
        
        ecg_data = load_data
        
        return ecg_data

def filter_ecg(read_data):
        ################KNN Imputer Filter##########################
        # Replace values below 500 as NaN   
        data = [x if x > 500 else np.nan for x in read_data]
        # Build Imputer 
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        # Fit Imputer to data
        data = imputer.fit_transform(np.reshape(data, (1, -1)))

        ############# #Low-Pass Filter##############################
        #Define lowpass filter
        lowpass = sps.butter(3, 0.1, btype='low', analog=False, output='ba')
        # Apply lowpass filter
        filtered_data = sps.filtfilt(*lowpass, data.flatten())

        #resample data to 6 sec * 360Hz
        filtered_data = sps.resample(filtered_data, 2160)

        return filtered_data

def predict_stress_heart(ecg_data):
        ###########Stress Detection################
        # Create HRV metric dictionaries
        HRV_data = hp.enhance_ecg_peaks(ecg_data, sample_rt)
        wd, m = hp.process(HRV_data, sample_rt, report_time=False, calc_freq=True)

        # Calculate HRV metrics of interest
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

        # Import model 
        stress_model = joblib.load('stress_model_v5.joblib')

        stress_label = stress_model.predict(HRV_metrics)

        return stress_label, HRV_metrics

def predict_stress_hrv(ecg_data):
        ###########Stress Detection################
        # Create HRV metric dictionaries

        m = get_time_domain_features(ecg_data)
        p = get_poincare_plot_features(ecg_data)

        # Calculate HRV metrics of interest
        MEAN_RR = m['mean_nni']
        MEDIAN_RR = m['median_nni']
        SDRR = m['sdnn']
        RMSSD = m['rmssd']
        SDSD = m['sdsd']
        HR = m['mean_hr']
        pNN50 = m['pnni_50']
        SD1 = p['sd1']
        SD2 = p['sd2']

        HRV_metrics = [[MEAN_RR, MEDIAN_RR, SDRR, RMSSD, SDSD,
                        HR, pNN50, SD1,SD2]]

        # Import model 
        stress_model = joblib.load('stress_model_v5.joblib')

        stress_label = stress_model.predict(HRV_metrics)

        return stress_label, HRV_metrics

def predict_arrhythmia(ecg_data):
        #############Arrhythmia Detection#######
        #Z-Score Data
        ecg_data = stats.zscore(ecg_data)
        ecg_data = ecg_data.reshape(ecg_data.shape[0], 1)

        arr_model = keras.models.load_model('arr_model_v4')

        arr_output = arr_model.predict(np.array([ecg_data,]), verbose=0)

        return arr_output


#Read unfiltered signal from Arduino
read_start = time.time_ns()
read_data = read_ecg()
read_end = time.time_ns()

'''
#Export list to csv for offline use
#df = pd.DataFrame(read_data)
#df.to_csv('list2.csv', index=False)

#Import csv to list for offline use 
read_data = []
with open('list1.csv', newline='') as inputfile:
    for row in csv.reader(inputfile):
        read_data.append(row[0])

read_data = list(map(int, read_data))
print(type(read_data))
print(len(read_data))
'''
#Filter Arduino signal
filter_start = time.time_ns()
ecg_data = filter_ecg(read_data)
filter_end = time.time_ns()

plt.figure()
plt.plot(read_data, color='orange', label="Input Signal")
plt.plot(ecg_data, label="Filtered Signal")
plt.legend(loc='upper right')
plt.xlabel('Sample number')
plt.ylabel('Amplitude (mV)')
#plt.title('ECG data filtered')
plt.show()

#Return probability of arrhythmia
arr_start = time.time_ns()
arr_label = predict_arrhythmia(ecg_data)
arr_end = time.time_ns()
print('Arrhythmia Confidence is:')
print(arr_label)

#Return stress classification and HRV metrics 
stress_hrv_start = time.time_ns()
stress_label, HRV_metrics = predict_stress_hrv(ecg_data)
stress_hrv_end = time.time_ns()
print('Stress Label using hrv-analysis is:')
print(stress_label)
print(HRV_metrics)

#stress_heart_start = time.time_ns()
#stress_label, HRV_metrics = predict_stress_heart(ecg_data)
#stress_heart_end = time.time_ns()
#print('Stress Label using HeartPy is:')
#print(stress_label)

print('')
print('Read time taken: ', read_end - read_start, 'ns')
print('Filter time taken: ', filter_end - filter_start, 'ns')
print('Arrhythmia time taken: ', arr_end - arr_start, 'ns')
print('Stress HRV time taken: ', stress_hrv_end - stress_hrv_start, 'ns')
#print('Stress HeartPy taken: ', stress_heart_end - stress_heart_start, 'ns')