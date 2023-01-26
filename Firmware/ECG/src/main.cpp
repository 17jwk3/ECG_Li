#include <Arduino.h>
// initialize the serial communication:
void setup(){
  Serial.begin(230400);
  pinMode(22, INPUT); // Setup for leads off detection LO +
  pinMode(23, INPUT); // Setup for leads off detection LO -
}

void loop() {
if((digitalRead(22) == 1)||(digitalRead(23) == 1)){
  Serial.println('!');
}
else{
// send the value of analog input 0:
  Serial.println(analogRead(26));
}
//Wait for a bit to keep serial data from saturating
  delay(1);
}