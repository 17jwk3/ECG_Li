#include <Arduino.h>
#include "BluetoothSerial.h"


#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Please run `make menuconfig` to and enable it
#endif

BluetoothSerial SerialBT;

void setup() {
  Serial.begin(115200);
  SerialBT.begin("ECG"); //Bluetooth device name
  Serial.println("The device started, now you can pair it with bluetooth!");
  pinMode(22, INPUT); // Setup for leads off detection LO +
  pinMode(23, INPUT); // Setup for leads off detection LO -
  pinMode(26, INPUT); // Setup for leads off detection LO -
}

void loop() {
  int data;
  data  = analogRead(26);
  // send the value of analog input 0:
  SerialBT.println(data);
  Serial.println(data);
  //Wait for a bit to keep serial data from saturating
  delay(1);
}
