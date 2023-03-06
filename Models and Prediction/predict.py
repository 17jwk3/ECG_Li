"""
Author: Alexander Koch, Ty Cymbalista, Jack Kay
@github: alexkoch14, tycym, 17jwk3
"""

##libaries 
import serial
#import serial.tools.list_ports
import numpy as np
import time
import heartpy as hp
import statistics
from scipy.signal import sps
from scipy import stats
from tensorflow import keras
from sklearn.impute import KNNImputer
import joblib
import warnings 
import os

# Global Parameters
num_sec = 3 #recording duration / 2
sample_rt = 360 #sample frequency
os.chdir("C:\\Users\\kocha\\Documents\\Queen's\\Year 4\\ELEC498\\ECG_Li\\Models and Prediction") #Path to project location
warnings.filterwarnings("ignore") #Ignore heartPy warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Ignore TensorFlow warnings



def read_ecg():
        #########Read ECG from Arduino#############
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

        return filtered_data

def predict_stress(ecg_data):
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

def predict_arrhythmia(ecg_data):
        #############Arrhythmia Detection#######
        #Z-Score Data
        ecg_data = stats.zscore(ecg_data)
        ecg_data = ecg_data.reshape(ecg_data.shape[0], 1)

        arr_model = keras.models.load_model('arr_model_v4')

        arr_output = arr_model.predict(np.array([ecg_data,]))

        return arr_output

#Read unfiltered signal from Arduino
read_data = read_ecg()
#Filter Arduino signal
ecg_data = filter_ecg(read_data)
#Return stress classification and HRV metrics 
stress_label, HRV_metrics = predict_stress(ecg_data)
#Return probability of arrhythmia
predict_arrhythmia(ecg_data)