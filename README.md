# Real-time Wearable-based Cardiac Monitoring with Machine Learning

## Design Problem
Heart disease describes a range of conditions that affect a person’s heart, including blood vessel disease, arrhythmias, valve disease, etc.Various tests are used to diagnose heart conditions. Doctors start with a risk assessment looking into medical history, blood pressure, as well as past and current symptoms but further tests may be required. These methods can be hard to access for most people, due to socio-economic factors. This is a long and outdated process in the digital era of telemedicine. Doctors and patients need fast, easily accessible procedures. 


Many forms of it can be prevented with healthy lifestyle choices and medical interventions. Stress can affect the body mentally and physically left unchecked manifests as an increased heart rate among other symptoms contributing to high blood pressure, diabetes, and heart disease.


Computational advancements have increased the ability to concurrently predict and prevent cardiac abnormalities. Thus creating faster, less intrusive, and more easily accessible methods for patients and practitioners alike.



## Design and Implementaiton

Low-cost ECG (electrocardiogram) electrodes using a 3 lead configuraiton are used to capture impulses through the heart muscles and transmit the information via Bluetooth to a machine-learning algorithm in real-time to provide actionable feedback on potential abnormalities and stress levels via a Graphical User Interface, as seen in the following implementation pipeline.

<img alt="Pipeline" src="media/Solution Pipeline.PNG" width="600"/>

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


#### Findings, Changes

The selected ESP32 module was a "standard" component for JLCPCB which means it costed much more to manufacture this board then originally anticipated. A chip that does not require "standard" assembly would be selected going forward. 

With more time, the AD8232 module would be integrated onto the board to increase portability with a very small lipo battery. Additional changes would be made to optimize the layout, add more LEDs for feedback etc. 

The output of the AD8232 may have needed to include a pulldown resistor. 

