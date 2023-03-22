# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer

from pathlib import Path

##libaries 
import serial
#import serial.tools.list_ports
import pandas as pd
import numpy as np
import time
from datetime import datetime
import heartpy as hp
import statistics
import scipy.signal as sps
from scipy import stats
from tensorflow import keras
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from hrvanalysis import get_time_domain_features, get_poincare_plot_features
import joblib
import warnings 
import os
# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
import tkinter as tk
# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage

# Global Parameters
num_sec = 3 #recording duration / 2
sample_rt = 360 #sample frequency
os.chdir("C:\\Users\\kocha\\Documents\\Queen's\\Year 4\\ELEC498\\ECG_Li\\Models and Prediction") #Path to project location
warnings.filterwarnings("ignore") #Ignore heartPy warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Ignore TensorFlow warnings
from pathlib import Path


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\\Users\\kocha\\Documents\\Queen's\\Year 4\\ELEC498\\ECG_Li\\Models and Prediction\\assets")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

window = Tk()

window.geometry("1200x720")
window.configure(bg = "#F1F1F1")

canvas = Canvas(
    window,
    bg = "#F1F1F1",
    height = 720,
    width = 1200,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    607.0,
    606.0,
    image=image_image_1
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    231.0,
    607.0,
    image=image_image_2
)

image_image_3 = PhotoImage(
    file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
    983.0,
    607.0,
    image=image_image_3
)

# =============================================================================
# canvas.create_rectangle(
#     880.0,
#     243.0,
#     1154.0,
#     274.0,
#     fill="#E95F8B",
#     outline="")
# =============================================================================

# =============================================================================
# canvas.create_rectangle(
#     880.0,
#     80.0,
#     1154.0,
#     216.0,
#     fill="#E95F8B",
#     outline="")
# =============================================================================

image_image_4 = PhotoImage(
    file=relative_to_assets("image_4.png"))
image_4 = canvas.create_image(
    231.0,
    517.0,
    image=image_image_4
)

canvas.create_text(
    546.0,
    571.0,
    anchor="nw",
    text="Stress Classifier",
    fill="#554CE1",
    font=("Inter Bold", 16 * -1)
)

canvas.create_text(
    163.0,
    571.0,
    anchor="nw",
    text="Arrythmia Detection",
    fill="#554CE1",
    font=("Inter Bold", 16 * -1)
)

canvas.create_text(
    916.0,
    571.0,
    anchor="nw",
    text="Advanced Analytics",
    fill="#554CE1",
    font=("Inter Bold", 16 * -1)
)

image_image_5 = PhotoImage(
    file=relative_to_assets("image_5.png"))
image_5 = canvas.create_image(
    924.0,
    380.0,
    image=image_image_5
)

image_image_6 = PhotoImage(
    file=relative_to_assets("image_6.png"))
image_6 = canvas.create_image(
    983.0,
    520.0,
    image=image_image_6
)

image_image_7 = PhotoImage(
    file=relative_to_assets("image_7.png"))
image_7 = canvas.create_image(
    605.0,
    518.0,
    image=image_image_7
)

image_image_8 = PhotoImage(
    file=relative_to_assets("image_8.png"))
image_8 = canvas.create_image(
    1114.0,
    29.0,
    image=image_image_8
)

image_image_9 = PhotoImage(
    file=relative_to_assets("image_9.png"))
image_9 = canvas.create_image(
    432.0,
    245.0,
    image=image_image_9
)

canvas.create_text(
    944.0,
    247.0,
    anchor="nw",
    text="See Full Report",
    fill="#FFFFFF",
    font=("Inter Bold", 18 * -1)
)

canvas.create_text(
    924.0,
    124.0,
    anchor="nw",
    text="Start Run ",
    fill="#FFFFFF",
    font=("Inter Bold", 40 * -1)
)

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

def predict_stress(ecg_data):
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

def main_plot(filtered_ecg):
    #importing Data
    #filtered_data = pd.read_csv('ECG-Test-2.csv')
    data = pd.DataFrame(filtered_ecg)
    figure1 = plt.Figure(figsize=(4, 2), dpi=100, facecolor='#554CE1')
    ax1 = figure1.add_subplot(111)
    ax1.patch.set_facecolor('#554CE1')
    ax1.spines['bottom'].set_color('#FFFFFF')
    ax1.spines['left'].set_color('#FFFFFF')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax1.set_xlabel('Timestamp (ms)', fontsize =14)
    ax1.set_ylabel('ECG Amplitude ()', fontsize =14)
    ax1.legend().set_visible(False)
    legend = ax1.legend()
    legend.remove()
    bar1 = FigureCanvasTkAgg(figure1, window)
    bar1.get_tk_widget().place(
        x=40,
        y=72.0,
        width=780,
        height=378.0)
    data.plot(kind='line', ax=ax1)

def BPM_Display(HRV_metrics):
    print("BPM Display")
    
    BPM = int(HRV_metrics[0][5])
    BPM_Text = (BPM, "BPM")
    
    canvas.create_text(
        1025.0,
        364.0,
        anchor="nw",
        text=(BPM_Text),
        fill="#E95F8B",
        font=("Inter SemiBold", 36 * -1)
        )

def advanced_analytics(HRV_metrics):
    ### NEED TO GO OVER THIS WITH ALEX
    #Mean RR
    MEAN_RR = int(HRV_metrics[0][0])
    RMSSD = int(HRV_metrics[0][3])
    SD1 = int(HRV_metrics[0][7])

    canvas.create_text(
        892.0,
        596.0,
        anchor="nw",
        text=("Mean RR:", MEAN_RR),
        fill="#554CE1",
        font=("Inter SemiBold", 14 * -1)
        )

    #RMSSD
    canvas.create_text(
        892.0,
        620.0,
        anchor="nw",
        text=("RMSSD:", RMSSD),
        fill="#554CE1",
        font=("Inter SemiBold", 14 * -1)
        )

    #SD1
    canvas.create_text(
        892.0,
        644.0,
        anchor="nw",
        text=("SD1:", SD1),
        fill="#554CE1",
        font=("Inter SemiBold", 14 * -1)
        )

def arrythmia_results(arr_output): 
    confidence = arr_output*100
    confidence_text = "Confidence:", confidence, "%"
    
    if arr_output < 0.5:
        result = "Heartbeat Normal"
        fill = "green"
        
    else: 
        result = "Heartbeat Abnormal"
        fill = "red"
    
    canvas.create_text(
        172.0,
        596.0,
        anchor="nw",
        text=result,
        fill=fill,
        font=("Inter Bold", 14 * -1)
        )    
    
    canvas.create_text(
        172.0,
        625.0,
        anchor="nw",
        text = (confidence_text),
        fill="#554CE1",
        font=("Inter Bold", 14 * -1)
        )
    
def stress_results(stress_label):
    if stress_label == 1:
        fill = "yellow"
        stress_text = 'Moderate Stress'
        
    elif stress_label == 2:
        fill = "red"
        stress_text = 'High Stress'
        
    else:
        fill = "green"
        stress_text = 'No Stress'
    
    canvas.create_text(
        545.0,
        596.0,
        anchor="nw",
        text=stress_text,
        fill=fill,
        font=("InterBold", 14 * -1)
        ) 

def runtime(run_time):
    run_time = (run_time, "s")
    canvas.create_text(
        1137.0,
        24.0,
        anchor="nw",
        text=(run_time),
        fill="#FFFFFF",
        font=("InterBold", 12 * -1)
        )

def update_fields(ecg_data, arr_output, stress_label, run_time, HRV_metrics):
    print("update fields has run")
    main_plot(ecg_data)
    advanced_analytics(HRV_metrics)
    arrythmia_results(arr_output)
    stress_results(stress_label)
    BPM_Display(HRV_metrics)
    runtime(run_time)
    
def function_runs():
    print("Function Runs")
    start = datetime.now()
    #start_time = now.strftime("%H:%M:%S")
    read_data = read_ecg()
    #Filter Arduino signal
    ecg_data = filter_ecg(read_data)
    #Return stress classification and HRV metrics 
    stress_label, HR = predict_stress(ecg_data)
    #Return probability of arrhythmia
    arr_output = predict_arrhythmia(ecg_data)
    end = datetime.now()
    
    #runtime calculations
    run_time = end - start 
    print(run_time)
    seconds = round(run_time.total_seconds(), 3)
    print(seconds)
    millis = seconds*1000
    print(millis)
    run_time = seconds
    
    #update the GUI Fields
    update_fields(ecg_data, arr_output, stress_label, run_time, HR)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_1 clicked"),
    relief="flat"
)
button_1.place(
    x=880.0,
    y=243.0,
    width=274.0,
    height=31.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command= lambda: function_runs(),
    relief="flat"
)
button_2.place(
    x=880.0,
    y=80.0,
    width=274.0,
    height=136.0
)

window.resizable(True, True)
window.mainloop()
