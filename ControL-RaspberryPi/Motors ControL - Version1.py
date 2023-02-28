import io
import socket
import struct
import time
import picamera
import RPi.GPIO as GPIO
import pygame
import numpy
import serial
from pygame.locals import *
import pygame.camera

mode = GPIO.getmode() 
GPIO.cleanup()

I1 = 32
I2 = 3
I3 = 33
I4 = 5
LED1 = 18
LED2 = 23
interept = 37

GPIO.setmode(GPIO.BOARD)

GPIO.setup(I1, GPIO.OUT)
GPIO.setup(I2, GPIO.OUT)
GPIO.setup(I3, GPIO.OUT)
GPIO.setup(I4, GPIO.OUT)
GPIO.setup(LED1, GPIO.OUT)
GPIO.setup(LED2, GPIO.OUT)
GPIO.setup(interept, GPIO.IN , pull_up_down = GPIO.PUD_UP)

pwm1=GPIO.PWM(I1,3000)
pwm3=GPIO.PWM(I3,3000)


def forward(x):
    pwm1.start(100-x)
    GPIO.output(I2,GPIO.HIGH)
    pwm3.start(100-x)
    GPIO.output(I4,GPIO.HIGH)
        
    #time.sleep(0.25)

    
def left(x):
    pwm1.start(100)
    GPIO.output(I2,GPIO.HIGH)
    pwm3.start(100-x)
    GPIO.output(I4,GPIO.HIGH)
    
    #time.sleep(0.25)
   

def right(x):
    pwm1.start(100-x)
    GPIO.output(I2,GPIO.HIGH)
    pwm3.start(100)
    GPIO.output(I4,GPIO.HIGH)
    
    #time.sleep(0.25)

def stop():
    
    pwm1.start(100)
    pwm3.start(100)
    #pwm1.stop()
    #pwm3.stop()
    #GPIO.output(I1,GPIO.HIGH)
    GPIO.output(I2,GPIO.HIGH)
    #GPIO.output(I3,GPIO.HIGH)
    GPIO.output(I4,GPIO.HIGH)
    

ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
ser.flush()

b = 0
ser_stop = dict()
ser_stop['change'] = 0

GPIO.output(LED1, GPIO.HIGH)
# time.sleep(0.25)

def aha(channel):
    stop()
    # print("int")
GPIO.add_event_detect(37,GPIO.FALLING , callback=aha , bouncetime=100)

while True:
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').rstrip()
        print(line)
        
        cls = line
        if cls == 'Looking center' or cls == 'Blinking':
            forward(50)
            b = 1
            ser_stop['change'] = 0
        elif cls == 'Looking left':
            right(50)
            b = 1
            ser_stop['change'] = 0
        elif cls == 'Looking right':
            left(50)
            b = 1
            ser_stop['change'] = 0
        elif cls == 'stop':
            stop()
            ser_stop['change'] = 0
        elif cls == 'Change' and b == 1:
            stop()
            ser_stop['change'] = ser_stop['change'] + 1
            if ser_stop['change'] == 2:
                break
        elif cls == 'Change' and b == 0:
            ser_stop['change'] = ser_stop['change'] + 1
            if ser_stop['change'] == 2:
                break
        

if b == 0:
    GPIO.output(LED1, GPIO.LOW)
    GPIO.output(LED2, GPIO.HIGH)
    print("Start Camera Control")
    s = socket.socket()
    host = '192.168.8.165'
    port = 8000
    s.connect((host, port))
    print('Connected')


    mp = dict()
    mp['stop'] = 0
    while True:
        if GPIO.input(interept) == 0:
            stop()
        try:
            cls = s.recv(1024).decode('utf-8')
            print(cls)
            
            if cls == 'stop':
                mp['stop'] += 1
                if mp['stop'] == 15:
                    break
            else:
                mp['stop'] = 0
                
            
            if cls == 'Looking center' or cls == 'Blinking':
                forward(50)
            elif cls == 'Looking left':
                right(50)
            elif cls == 'Looking right':
                left(50)
            elif cls == 'stop':
                stop()
        except:
            break

if b == 0:
    s.close()
#pwm1.stop()
#pwm3.stop()
GPIO.output(LED1, GPIO.LOW)
GPIO.output(LED2, GPIO.LOW)
GPIO.cleanup()
print("bye")
