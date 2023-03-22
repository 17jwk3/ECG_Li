# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:23:50 2022

@author: tycym
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import os
import warnings
import scipy.signal as sps
import matplotlib.pyplot as plt
#from bluetooth import *


# Global Parameters
num_sec = 3 #recording duration / 2
sample_rt = 360 #sample frequency
os.chdir("C:\\Users\\kocha\\Documents\\Queen's\\Year 4\\ELEC498\\ECG_Li") #Path to project location
warnings.filterwarnings("ignore") #Ignore heartPy warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Ignore TensorFlow warnings



# Import data
#ecg_data = np.genfromtxt('ECG Samples\\Front Delt - Iliac Crest.csv', delimiter=',')
ecg_data = np.genfromtxt('ECG Samples\\Upper Chest - Rib Cage Dropped Values.csv', delimiter=',')





################KNN Imputer Filter##########################
# Replace values below 500 as NaN   
data = [x if x > 500 else np.nan for x in ecg_data]
# Build Imputer 
imputer = KNNImputer(n_neighbors=5, weights='distance')
# Fit Imputer to data
data = imputer.fit_transform(np.reshape(data, (1, -1)))





#################Low Pass Filter########################## 
# Define filter coefficients for lowpass, bandpass, and highpass filters
lowpass = sps.butter(3, 0.1, btype='low', analog=False, output='ba')
#bandpass = sps.butter(3, [0.1, 0.5], btype='band', analog=False, output='ba')
#highpass = sps.butter(3, 0.01, btype='high', analog=False, output='ba')

# Apply lowpass filter
data = sps.filtfilt(*lowpass, data.flatten())

# Plot original ECG data and filtered data
plt.figure()
plt.plot(ecg_data, color='orange', label="Input Signal")
plt.plot(data, label="Filtered Signal")
plt.legend(loc='upper right')
plt.xlabel('Sample number')
plt.ylabel('Amplitude (mV)')
#plt.title('ECG data filtered')

plt.show()