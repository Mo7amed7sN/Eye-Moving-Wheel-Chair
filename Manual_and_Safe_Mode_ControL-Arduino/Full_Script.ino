#define joyX   A0
#define joyY A1
int sw ;    
int xMap, yMap, xValue, yValue;
int a=0;
int trigPin1 = 9;
int echoPin1 = 10;
int trigPin2 = 5;
int echoPin2 = 6;
long duration1;
int distance1;
int safetyDistance;
long duration2;
int distance2;
int interupt1 = 12;
void setup() {
  Serial.begin(9600);
  pinMode(7,INPUT);
  pinMode(trigPin1, OUTPUT); 
  pinMode(echoPin1, INPUT);
  pinMode(trigPin2, OUTPUT); 
  pinMode(echoPin2, INPUT);
  pinMode(interupt1, OUTPUT);
  digitalWrite(interupt1, HIGH); 
}
 
void loop() {
  // put your main code here, to run repeatedly:
  digitalWrite(trigPin1, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin1, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin1, LOW);
  duration1 = pulseIn(echoPin1, HIGH);
  distance1= duration1*0.034/2;
  
  digitalWrite(trigPin2, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin2, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin2, LOW);
  duration2 = pulseIn(echoPin2, HIGH);  
  distance2= duration2*0.034/2;
  //Serial.println(distance1);
  //Serial.println(distance2);
/*
  if (distance1 <= 25 && distance2 <= 25){
  //Serial.println("Change");
  digitalWrite(interupt1, LOW);  
}*/
 if (distance1 <= 100){
  //Serial.println("Change");
  digitalWrite(interupt1, LOW);  
}
 else if (distance2 <= 100){
  //Serial.println("Change");
  digitalWrite(interupt1, LOW);  
} 
else{
  digitalWrite(interupt1, HIGH);
  xValue = analogRead(joyX);
  yValue = analogRead(joyY);
  xMap = map(xValue, 0,1023, 0, 7);
  yMap = map(yValue,0,1023,7,0);
  sw = analogRead(A2);
  if(xMap==7 && yMap==4){
    Serial.println("Looking center");
    a=0;
  }
  else if(sw==0) {
    delay(100);
    if(sw==0){
    a++;
    if(a==2){
    Serial.println("Change");
    a=0;
    }
    }
  }
  else if(xMap==3 && yMap==4){
    Serial.println("stop");
    a=0;}
  else if(xMap==3 && yMap<=1){            
    Serial.println("Looking right");
    a=0;
  }
  else if(xMap==3 && yMap==7){
    Serial.println("Looking left");  
  }
}
delay(100);
}
