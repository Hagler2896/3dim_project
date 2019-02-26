#include <SoftwareSerial.h>

SoftwareSerial HC11(2, 3);

#define S0 4
#define S1 5
#define S2 6
#define S3 7
#define sensorOut 12

#define M_La 8
#define M_Lb 9
#define M_Ra 10
#define M_Rb 11

char rxDataBuf[10];
boolean start_code = false;
int drive_status = -1;
int drive_angle = 0;
int rxCount = 0;
int spd = 150;

int redFrequency = 0;
int greenFrequency = 0;
int blueFrequency = 0;

int red = 0;
int green = 0;
int blue = 0;

void setup(){
  pinMode(M_La, OUTPUT);
  pinMode(M_Lb, OUTPUT);
  pinMode(M_Ra, OUTPUT);
  pinMode(M_Rb, OUTPUT);
  Serial.begin(9600);
  pinMode(S0, OUTPUT);
  pinMode(S1, OUTPUT);
  pinMode(S2, OUTPUT);
  pinMode(S3, OUTPUT);

  // Setting the sensorOut as an input
  pinMode(sensorOut, INPUT);

  // Setting frequency scaling to 20%
  digitalWrite(S0, HIGH);
  digitalWrite(S1, LOW);

  HC11.begin(9600);
}

void loop(){
  
    digitalWrite(S2, LOW);
  digitalWrite(S3, LOW);

  // Reading the output frequency
  redFrequency = pulseIn(sensorOut, LOW);
  // Remaping the value of the RED (R) frequency from 0 to 255
  // You must replace with your own values. Here's an example:
  // redColor = map(redFrequency, 70, 120, 255,0);
  red = map(redFrequency, 18, 255, 255, 0);
  
    digitalWrite(S2, HIGH);
  digitalWrite(S3, HIGH);

  // Reading the output frequency
  greenFrequency = pulseIn(sensorOut, LOW);
  // Remaping the value of the GREEN (G) frequency from 0 to 255
  // You must replace with your own values. Here's an example:
  // greenColor = map(greenFrequency, 100, 199, 255, 0);
  green = map(greenFrequency, 93, 383, 255, 0);
  
    digitalWrite(S2, LOW);
  digitalWrite(S3, HIGH);

  // Reading the output frequency
  blueFrequency = pulseIn(sensorOut, LOW);
  // Remaping the value of the BLUE (B) frequency from 0 to 255
  // You must replace with your own values. Here's an example:
  // blueColor = map(blueFrequency, 38, 84, 255, 0);
  blue = map(blueFrequency, 24, 231, 255, 0);
  Serial.print(red);
  Serial.print(" ");
  Serial.print(green);
  Serial.print(" ");
  Serial.println(blue);
    if (red > green && red > blue) {
    Serial.println("  RED detected!");
    drive_status = 4;
  }
  if ((red > 145 && red < 200) &&  170 < blue) {
    Serial.println("  GREEN_2");
    HC11.write('#');
    HC11.write('2');
    HC11.write('@');

  }
  if((red > 90 && red < 140) && (green > 200 && green < 240) && (blue > 195 && blue < 230)){
    Serial.println("   BLUE_1");
    HC11.write('#');
    HC11.write('1');
    HC11.write('@');
  }
  
  if(Serial.available()){
    char c = Serial.read();
    Serial.println(c);
   if(start_code == false){
    if(c == '#')  start_code = true;
   }else{
    if(c == '@'){
     dataParsing(rxDataBuf); 
     start_code = false;
     rxCount=0;
     memset(rxDataBuf,0,10);
    }
    else{
      rxDataBuf[rxCount++] = c;
      }
    }
  }
  drive_control(drive_status);
}

void dataParsing(char* buf){
  Serial.println("parsing");
  drive_status = (buf[0] - '0');
  drive_angle = (buf[1] - '0');
  Serial.print("drive_status: ");
  Serial.println(drive_status);
  Serial.print("drive_angle: ");
  Serial.print(drive_angle);
}

void drive_control(int mode){
  if(mode == 0){
      Serial.println("Forward");
      digitalWrite(M_La,LOW);
      analogWrite(M_Lb,spd);
      digitalWrite(M_Ra,LOW);
      analogWrite(M_Rb,spd);
    }
  else if(mode == 1){
    Serial.println("Back");
    analogWrite(M_La,spd);
    digitalWrite(M_Lb,LOW);
    analogWrite(M_Ra,spd);
    digitalWrite(M_Rb,LOW);
  }
  else if(mode == 2){
    Serial.println("Left");
    digitalWrite(M_La,LOW);
    analogWrite(M_Lb,75);
    analogWrite(M_Ra,75);
    digitalWrite(M_Rb,LOW); 
    
  }
  else if(mode == 3){
    Serial.println("Right");
    analogWrite(M_La,75);
    digitalWrite(M_Lb,LOW);
    digitalWrite(M_Ra,LOW);
    analogWrite(M_Rb,75);  
  }
  else if(mode == 4){
    Serial.println("stop");
    digitalWrite(M_La,LOW);
    digitalWrite(M_Lb,LOW);
    digitalWrite(M_Ra,LOW);
    digitalWrite(M_Rb,LOW); 
  }
}
