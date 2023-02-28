import io
import socket
import struct
import time
from picamera.array import PiRGBArray
from picamera import PiCamera
import RPi.GPIO as GPIO
import numpy as np
import serial
from gaze_tracking import GazeTracking
from PIL import Image
import cv2
import Models


gaze = GazeTracking()


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


camera = PiCamera()
camera.vflip = True
camera.resolution = (512, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(512, 480))


ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
ser.flush()

def forward(x):
    pwm1.start(100 - x)
    GPIO.output(I2, GPIO.HIGH)
    pwm3.start(100 - x)
    GPIO.output(I4, GPIO.HIGH)

    # time.sleep(0.25)


def left(x):
    pwm1.start(100)
    GPIO.output(I2, GPIO.HIGH)
    pwm3.start(100 - x)
    GPIO.output(I4, GPIO.HIGH)

    # time.sleep(0.25)

def right(x):
    pwm1.start(100 - x)
    GPIO.output(I2, GPIO.HIGH)
    pwm3.start(100)
    GPIO.output(I4, GPIO.HIGH)

    # time.sleep(0.25)

def stop():
    pwm1.start(100)
    pwm3.start(100)
    # pwm1.stop()
    # pwm3.stop()
    # GPIO.output(I1,GPIO.HIGH)
    GPIO.output(I2, GPIO.HIGH)
    # GPIO.output(I3,GPIO.HIGH)
    GPIO.output(I4, GPIO.HIGH)


def aha(channel):
    stop()
    # print("int")
GPIO.add_event_detect(37,GPIO.FALLING , callback=aha , bouncetime=100)


while True:
    GPIO.output(LED2, GPIO.HIGH)
    GPIO.output(LED1, GPIO.LOW)
    ser_stop = dict()
    ser_stop['change'] = 0
    b = 0
    while True:
        if GPIO.input(37) == False:
            stop()
            continue
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()
            print(line)

            cls = line
            if cls == 'Looking center' or cls == 'Blinking':
                forward(50)
                b = 1
                ser_stop['change'] = 0
            elif cls == 'Looking left':
                left(50)
            elif cls == 'Looking right':
                right(50)
                b = 1
                ser_stop['change'] = 0
            elif cls == 'stop':
                stop()
                ser_stop['change'] = 0
            elif cls == 'Change' and b == 1:
                b = 0
                stop()
                ser_stop['change'] = ser_stop['change'] + 1
                if ser_stop['change'] == 2:
                    break
            elif cls == 'Change' and b == 0:
                ser_stop['change'] = ser_stop['change'] + 1
                if ser_stop['change'] == 2:
                    break
            

    if b == 0:
        GPIO.output(LED2, GPIO.LOW)
        GPIO.output(LED1, GPIO.HIGH)

        print("Start Camera Control")
        
        'camera.start_preview()'
        time.sleep(0.25)

        start = time.time()
        stream = io.BytesIO()

        cls = 'E'
        tmp = 'E'
        mp = dict()
        mp['stop'] = 0
        br = False

        for rgb_im in camera.capture_continuous(rawCapture, format = "bgr", use_video_port = True):
            if GPIO.input(37) == False:
                stop()
                rawCapture.truncate(0)
                rawCapture.seek(0)
                continue

            rgb_im = rgb_im.array
            # print(rgb_im)

            try:
                gaze.refresh(rgb_im)
                rgb_im = gaze.annotated_frame()
                cls = Models.Get_Direction(gaze)
            except:
                pass


            if gaze.eye_left == None and gaze.eye_right == None:
                cls = 'stopC'
                mp['stop'] += 1
                if mp['stop'] == 5 and br == True:
                    stop()
                    stop()
                    rawCapture.truncate(0)
                    rawCapture.seek(0)
                    break
            else:
                mp['stop'] = 0

            if cls is None:
                cls = tmp
            else:
                tmp = cls


            if cls == 'F' or cls == 'B':
                forward(32)
                br = True
            elif cls == 'L':
                right(32)
                br = True
            elif cls == 'R':
                left(32)
                br = True
            elif cls == 'stop':
                stop()

            rawCapture.truncate(0)
            rawCapture.seek(0)

            print(cls)

            '''cv2.putText(rgb_im, cls, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
            left_pupil = gaze.pupil_left_coords()
            right_pupil = gaze.pupil_right_coords()

            cv2.putText(rgb_im, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 3$            cv2.putText(rgb_im, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, $
            cv2.imshow("Demo", rgb_im)

            if cv2.waitKey(1) == 27:
                break
            '''



    GPIO.output(LED1, GPIO.LOW)
    GPIO.output(LED2, GPIO.LOW)
    time.sleep(0.25)

    # GPIO.cleanup()
    # pwm1.stop()
    # pwm3.stop()
            
GPIO.output(LED1, GPIO.LOW)
GPIO.output(LED2, GPIO.LOW)
GPIO.cleanup()
