# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:23:50 2022

@author: tycym
"""

##libaries 
import string
import serial
import serial.tools.list_ports
import numpy as np
import pandas as pd
import time
#import ecg_plot
#from bluetooth import *
#from sklearn.impute import KNNImputer

#declare variables 
global load_data, sers
global baudrate

load_data = []
baudrate = 230400
processed_data = []
test =[]

## COM port setup
sers = {}
key = 'COM7'
sers[key] = serial.Serial(key, baudrate)

##COM to Data
pf = 230400 #polling freqeuncy 
duration = 10 

t_end = time.time() + 6
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
    


#create dataframe of the data with column names Time, mV
data = pd.DataFrame(load_data, columns=['mV'])

#export to csv
data.to_csv('ECG_reciver_test.csv')

#close the serial
serial.Serial(key, baudrate).close()