# Real-time Wearable-based Cardiac Monitoring with Machine Learning

## Design Problem
Heart disease describes a range of conditions that affect a person’s heart, including blood vessel disease, arrhythmias, valve disease, etc.Various tests are used to diagnose heart conditions. Doctors start with a risk assessment looking into medical history, blood pressure, as well as past and current symptoms but further tests may be required. These methods can be hard to access for most people, due to socio-economic factors. This is a long and outdated process in the digital era of telemedicine. Doctors and patients need fast, easily accessible procedures. 


Many forms of it can be prevented with healthy lifestyle choices and medical interventions. Stress can affect the body mentally and physically left unchecked manifests as an increased heart rate among other symptoms contributing to high blood pressure, diabetes, and heart disease.


Computational advancements have increased the ability to concurrently predict and prevent cardiac abnormalities. Thus creating faster, less intrusive, and more easily accessible methods for patients and practitioners alike.



## Design and Implementaiton

Low-cost ECG (electrocardiogram) electrodes using a 3 lead configuraiton are used to capture impulses through the heart muscles and transmit the information via Bluetooth to a machine-learning algorithm in real-time to provide actionable feedback on potential abnormalities and stress levels via a Graphical User Interface, as seen in the following implementation pipeline.

<img alt="pipeline" src="media/pipeline.png" width="600"/>

---------------

### Hardware @17jwk3

The PCB uses a ESP32-C3-MINI-1-N4, a Li-Po with a battery charging module and a buck conmverter to supply 3.3V to the rest of the components. 
The main ECG components are an AD8232 opAmp board utilized for reading the electrical signals of the wearers heart and jack port for the 3 leads. 


#### PCB
This is a 4 layer PCB board that uses a AD8232 breakout board for the ECG pickup. 
The schematic, featuring over charge and discharge protection, switches to control: MCU power, 5V (charging) and Battery power. Also includes ESD protection for usb type C. The brains is an ESP32-C3-MINI-1-N4.

<img alt="3D model" src="media/pcb_3d_1.PNG" width="600"/>
<img alt="4 layers" src="media/pcb_3d_2.PNG" width="600"/>

#### PCB layout 
<img alt="Layer 1,2,3,4" src="media/pcb_1.PNG" width="600"/>
<img alt="Layer 2,3,4" src="media/pcb_2.PNG" width="600"/>
<img alt="Layer 3,4" src="media/pcb_3.PNG" width="600"/>
<img alt="Layer 4" src="media/pcb_4.PNG" width="600"/>

#### The schematic
<img alt="Layer 4" src="media/schematic.PNG" width="600"/>


#### Findings & Changes

The selected ESP32 module was a "standard" component for JLCPCB which means it costed much more to manufacture this board then originally anticipated. A chip that does not require "standard" assembly would be selected going forward. 

With more time, the AD8232 module would be integrated onto the board to increase portability with a very small lipo battery. Additional changes would be made to optimize the layout, add more LEDs for feedback etc. 

The output of the AD8232 may have needed to include a pulldown resistor. 


---------------

### Data Acquisition @17jwk3
#### Design

#### Findings & Changes


---------------

### Processing and Filtering @CTy27
#### Design

#### Findings & Changes


---------------

### Arrhythmia Detection @alexkoch14
#### Design
The arrhythmia model was trained on the MIT-BIH database, containing 48-half-hour excerpts of two-channel ambulatory ECG recordings, obtained from 47 subjects studied by the BIH arrhythmia laboratory between 1975 and 1979.

The recordings were digitized at 360 samples per second and two or more cardiologists independently annotated each record.

The recordings were extracted using wfdb, the Python wave-form database package. ECG signals are transformed into snippets during the data pre-processing step as the model analyzes one-dimensional input data. 

Heartbeats in the first or last 3 seconds of the half-hour recording are ignored to reduce data clipping sizes, as is for any non-beat annotations. Beat symbols are encoded as abnormal (1) or normal (0). 

A sliding window of 6 seconds per beat segment is used to capture the current beat and those adjacent to retain enough information from encompassed patterns ensuring a 2-3 beat overlap.

Heartbeat voltages range among individuals depending on anatomy, body composition, blood volume, and ECG lead placement. Z-score normalization standardizes values into how much each reading deviates from the mean of all readings within the sample window

In order to create the dataset locally, run 'arrhythmia_data.py'.

<img alt="arr_data_comparison" src="media/arr_data_comparison.png" width="600"/>

<img alt="CNN" src="media/CNN_v4.png" width="600"/>

Leave-One-Out Crass-validation was employed via a hold-out set. The database was divided into three sets using a 60/30/10 train/validation/test split.
The holdout set primarily sought to encompass ~5 unseen subjects, consequentially representing ~10% of the data. 

The Deep-Learning (DL) model follows a Convolutional Neural Network architecture (CNN) using a TensorFlow backend

CNN_V4 uses a shallow, wide and simple architecture to overcome common real-time classification issues. 

The dataset has few features, but lots of records. This is perfect for DL models that need fewer parameters and sample complexity to achieve acceptable performance.

The model was trained on three passings of the training data (epochs). Accuracy decreased, loss increased, and overfitting began to occur beyond this point. 

In order to replicate the model, run 'arrhythmia_model.py'

<img alt="CNN_loss" src="media/CNN_v4_acc_loss.png" width="600"/>


#### Findings & Changes
CNN v4’s performance indicates 87.7% accuracy and 84% recall on the test set, which comprises unseen patients, as seen in media/CNN_v4_results.txt

This is satisfactory for real-time classification uses which aren’t meant to serve as a clinical diagnosis. 

It showed minimal signs of overfitting with training loss and validation losses of 0.2195 and 0.2264 respectively. 

<img alt="CNN_v4_results" src="media/CNN_v4_results.png" width="600"/>

A limitation of the current methodology lies in grouping all abnormal patterns to obtain even distribution without resampling. It is possible that CNN v4’s ~90% accuracy is accredited is predicting 80% of abnormality types with 100% accuracy, and the remaining 20% with 0% accuracy. The model might not be able to predict less prevalent abnormality patterns while being excellent in predicting common ones.

---------------

### Stress Detection @alexkoch14
#### Design
The multimodal SWELL Knowledge Work dataset for stress modeling (SWELL-KW) contains data from 25 participants (~3 hours each) performing typical office work (writing reports, reading emails, etc.) under 3 conditions: no stress (0), email interruptions (1) and time pressure (2).

The stress model was trained on HRV metrics obtained from the SWELL-KW dataset.
Inter-beat interval (IBI) samples are extracted from the raw ECG signal of each subject over the 3-hour time series. 
HRV indices are then computed on a subset of the IBI signal array. New samples are appended to the IBI array while the oldest is removed from the beginning,creating a sliding window of HRV indices over a fixed acquisition window analogous to the arrhythmia data. 
This facilitates a granular study of how momentary heartbeat patterns and concurrent HRV metrics reflect a subject’s stress level. 

HRV metrics are computed from the input filtered signal using hrv-analysis Python library.

Leave-One-Out Crass-validation was employed via a hold-out set. The database was divided into three sets using a 60/30/10 train/validation/test split.
The holdout set primarily sought to encompass ~3 unseen subjects, consequentially representing ~10% of the data. 

Twenty-three classification models were then assessed using the LazyPredict Python library, indicating Guassian Naive Bayes provides the best time/accuracy trade-off.

<img alt="stress_comparison" src="media/stress_comparison.png" width="600"/>

#### Findings & Changes

Stress model performance indicates a 57.1% F1-score on the validaiton set and 67.6% F1-Score on the test set, as seen in media/stress_model_v5_results.txt

<img alt="stress_model_v5_results" src="media/stress_model_v5_results.png" width="600"/>

The disparity in validation and testing sets accentuates some underlying points of issue. 

Models possess inherent predictive error which may lead to a relatively higher score on the test set due to chance.

The current neurobiological evidence suggests that HRV is impacted by stress and supports its use for an objective assessment of health, but it can be expressed through other psychological pathways (EDA, body posture, etc.).
These added features, which are not part of the feature set, are perhaps expressed more vividly in the validation subjects. 
Equivalently said, fewer people in the validation set tend to manifest stress predominantly through HRV than other physiological pathways. This is a limitation of training the stress model on a subset of a subset of the SWELL-KW dataset. 

Nonetheless, results are satisfactory for obtaining real-time assessments. Stress mitigation is a preventative step in cardiac abnormality evasion. It is the first line of defense, but an overpassed subject will get caught using CNN v4 used in arrhythmia detection.  

---------------

### Graphical User Interface @CTy27
#### Design


#### Findings & Changes


---------------

## Replication 

'ECG_Li\Models and Prediction\gui.py'
