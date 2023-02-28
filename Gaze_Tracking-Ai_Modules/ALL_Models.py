import cv2
import numpy as np
from numpy import zeros, newaxis
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout

from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras import optimizers
import operator
from difflib import SequenceMatcher
import speech_recognition as sr

def Load_Model2():
    eye_landmarks_model = CNN_Model_Landmark()
    Left_Eye_Gaze_Model = CNN_Model_Eye_Gaze()
    Right_Eye_Gaze_Model = CNN_Model_Eye_Gaze()

    # load the best weight to the model
    eye_landmarks_model.load_weights('G:\wheel chair\FullFace/landmark_weights.final.hdf5')
    Left_Eye_Gaze_Model.load_weights('G:\wheel chair\FullFace/weights_left_eye_gaze.final.hdf5')
    Right_Eye_Gaze_Model.load_weights(
        'G:\wheel chair\FullFace/weights_right_eye_gaze.final.hdf5')

    return eye_landmarks_model, Left_Eye_Gaze_Model, Right_Eye_Gaze_Model


def Load_Model():
    face_cascade = cv2.CascadeClassifier('G:\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('G:\opencv-master\data\haarcascades\haarcascade_eye_tree_eyeglasses.xml')

    pixels = 60
    CNN_Model = Sequential([Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(pixels, pixels, 1)),
                            MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'),
                            Dropout(0.25),
                            Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
                            MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'),
                            Dropout(0.25),
                            Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
                            MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'),
                            Dropout(0.25),
                            Flatten(),
                            Dense(400, activation='relu'),
                            Dropout(0.25),
                            Dense(4, activation='softmax')
                            ])

    CNN_Model.load_weights('G:\wheel chair\Proctoring-AI-master\eye_tracking\weights.h5')

    return CNN_Model, face_cascade, eye_cascade


def Sign(key):
    ret = None
    if key == 0:
        ret = 'C'
    elif key == 1:
        ret = 'L'
    elif key == 2:
        ret = 'R'
    elif key == 3:
        ret = 'F'

    return ret


def Detect_Eye(img, face_cascade, eye_cascade):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    img1 = None
    for (x, y , w ,h) in faces:
        'cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0 , 0), 3)'
        roi_gray = gray[y:y+h, x:x+w]
        'roi_color = img[y:y+h, x:x+w]'
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey ,ew, eh) in eyes:
            'cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 5)'
            img1 = roi_gray[ey:ey+eh, ex:ex+ew]

    '''cv2.imshow('ss', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    return img1


def Classify_Direction(img, CNN_Model):
    img1 = cv2.resize(img, (60, 60))
    img1 = img1[:, :, newaxis]
    img1 = np.expand_dims(img1, axis=0)

    pred = np.argmax(CNN_Model.predict(np.array(img1))[0])
    print(pred)

    return Sign(pred)


def Get_Direction(gaze):
    ret = None

    if gaze.is_blinking():
        ret = "B"
    elif gaze.is_right():
        ret = "R"
    elif gaze.is_left():
        ret = "L"
    elif gaze.is_center():
        ret = "F"

    return ret


def face_extraction(image):
    # image = cv2.imread(img_path)

    # Convert the image to RGB colorspace
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert the RGB  image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Extract the pre-trained face detector from an xml file
    face_cascade = cv2.CascadeClassifier(
        'G:\wheel chair\FullFace/haarcascade_frontalface_default.xml')

    # Detect the faces in image
    faces = face_cascade.detectMultiScale(gray, 1.25, 6)

    # Make a copy of the orginal image to draw face detections on
    image_with_detections = np.copy(image)
    (x, y, w, h) = faces[0]

    gray = gray[y:y + h, x:x + w]
    gray = cv2.resize(gray, (96, 96)) / 255

    return gray, (x, y, w, h)


def get_eyes(prediction, img, face_details):
    (x, y, w, h) = face_details
    left_eye_inner_corner_x = (prediction[4] * w / 96) + x
    left_eye_inner_corner_y = (prediction[5] * h / 96) + y
    left_eye_outer_corner_x = (prediction[6] * w / 96) + x
    left_eye_outer_corner_y = (prediction[7] * h / 96) + y
    right_eye_inner_corner_x = (prediction[8] * w / 96) + x
    right_eye_inner_corner_y = (prediction[9] * h / 96) + y
    right_eye_outer_corner_x = (prediction[10] * w / 96) + x
    right_eye_outer_corner_y = (prediction[11] * h / 96) + y
    left_eyebrow_inner_end_x = (prediction[12] * w / 96) + x
    left_eyebrow_inner_end_y = (prediction[13] * h / 96) + y
    left_eyebrow_outer_end_x = (prediction[14] * w / 96) + x
    left_eyebrow_outer_end_y = (prediction[15] * h / 96) + y
    right_eyebrow_inner_end_x = (prediction[16] * w / 96) + x
    right_eyebrow_inner_end_y = (prediction[17] * h / 96) + y
    right_eyebrow_outer_end_x = (prediction[18] * w / 96) + x
    right_eyebrow_outer_end_y = (prediction[19] * h / 96) + y
    height_diff_left_inner = abs(left_eyebrow_inner_end_y - left_eye_inner_corner_y)
    height_diff_left_outer = abs(left_eyebrow_outer_end_y - left_eye_outer_corner_y)
    height_left_eye = max(height_diff_left_inner, height_diff_left_outer) * 2
    height_diff_right_inner = abs(right_eyebrow_inner_end_y - right_eye_inner_corner_y)
    height_diff_right_outer = abs(right_eyebrow_outer_end_y - right_eye_outer_corner_y)
    height_right_eye = max(height_diff_right_inner, height_diff_right_outer) * 2
    if abs(left_eyebrow_outer_end_x - left_eyebrow_inner_end_x) > abs(
            left_eye_outer_corner_x - left_eye_inner_corner_x):
        width_left_eye = abs(left_eyebrow_outer_end_x - left_eyebrow_inner_end_x)
        left_eye_x = min(left_eyebrow_outer_end_x, left_eyebrow_inner_end_x)
    else:
        width_left_eye = abs(left_eye_outer_corner_x - left_eye_inner_corner_x)
        left_eye_x = min(left_eye_outer_corner_x, left_eye_inner_corner_x)
    if abs(right_eyebrow_outer_end_x - right_eyebrow_inner_end_x) > abs(
            right_eye_outer_corner_x - right_eye_inner_corner_x):
        width_right_eye = abs(right_eyebrow_outer_end_x - right_eyebrow_inner_end_x)
        right_eye_x = min(right_eyebrow_outer_end_x, right_eyebrow_inner_end_x)
    else:
        width_right_eye = abs(right_eye_outer_corner_x - right_eye_inner_corner_x)
        right_eye_x = min(right_eye_outer_corner_x, right_eye_inner_corner_x)

    #  img = Image.open(folderpath)
    img = Image.fromarray(np.uint8(img)).convert('RGB')

    left_eye_rect = (
    left_eye_x, left_eyebrow_outer_end_y, left_eye_x + width_left_eye, left_eyebrow_outer_end_y + height_left_eye)
    left_eye_img = img.crop(left_eye_rect).convert('L')
    left_eye_img = left_eye_img.resize((50, 42), Image.ANTIALIAS)
    right_eye_rect = (
    right_eye_x, right_eyebrow_inner_end_y, right_eye_x + width_right_eye, right_eyebrow_inner_end_y + height_right_eye)
    right_eye_img = img.crop(right_eye_rect).convert('L')
    right_eye_img = right_eye_img.resize((50, 42), Image.ANTIALIAS)
    return left_eye_img, right_eye_img


def CNN_Model_Landmark():
    model = Sequential()

    model.add(Conv2D(
        input_shape=(96, 96, 1),
        filters=16,
        kernel_size=(3, 3),
        strides=1,
        padding='valid',
        activation='relu',
        data_format='channels_last'
    ))

    model.add(MaxPooling2D(
        pool_size=(2, 2),
    ))

    model.add(Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu',
        strides=1,
        padding='valid',
    ))

    model.add(MaxPooling2D(
        pool_size=(2, 2),
    ))

    model.add(Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu',
        strides=1,
        padding='valid',
    ))

    model.add(MaxPooling2D(
        pool_size=(2, 2),
    ))

    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu',
        strides=1,
        padding='valid',
    ))

    model.add(MaxPooling2D(
        pool_size=(2, 2),
    ))

    model.add(Flatten())

    model.add(Dense(
        units=512,
        activation='relu'
    ))

    model.add(Dropout(
        rate=0.2
    ))

    model.add(Dense(
        units=30,
    ))

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    # model.summary()

    return model


def CNN_Model_Eye_Gaze():
    model = Sequential()

    model.add(Conv2D(
        input_shape=(42, 50, 1),
        filters=24,
        kernel_size=(7, 7),
        strides=1,
        padding='valid',
        activation='relu',
        data_format='channels_last'
    ))

    model.add(MaxPooling2D(
        pool_size=(2, 2),
    ))

    model.add(Conv2D(
        filters=24,
        kernel_size=(5, 5),
        activation='relu',
        strides=1,
        padding='valid',
    ))

    model.add(MaxPooling2D(
        pool_size=(2, 2),
    ))

    model.add(Conv2D(
        filters=24,
        kernel_size=(3, 3),
        activation='relu',
        strides=1,
        padding='valid',
    ))

    model.add(MaxPooling2D(
        pool_size=(2, 2),
    ))

    model.add(Flatten())

    model.add(Dense(
        units=7,
        activation='softmax'
    ))

    sgd_optimizer = optimizers.SGD(
        lr=0.1, decay=1e-6, nesterov=True, momentum=0.9)

    model.compile(optimizer=sgd_optimizer, loss='mse', metrics=['accuracy'])

    # model.summary()

    return model


def final_prediction(left_eye_prob, right_eye_prob):
    overall_prob = (left_eye_prob + right_eye_prob) / 2
    index, value = max(enumerate(overall_prob), key=operator.itemgetter(1))
    return index


def predict(img, eye_landmarks_model, Left_Eye_Gaze_Model, Right_Eye_Gaze_Model):
    class_name = ['Center', 'Down Left', 'Down Right', 'Left', 'Right', 'Up Left', 'Up Right']
    face_img, (x, y, w, h) = face_extraction(img)
    landmarks_prediction = np.squeeze(
        eye_landmarks_model.predict(np.expand_dims(np.expand_dims(face_img, axis=-1), axis=0)))
    landmarks_prediction = landmarks_prediction * 48 + 48
    left_eye_img, right_eye_img = get_eyes(landmarks_prediction, img, (x, y, w, h))
    left_eye_img.load()
    img_arr_left = np.asarray(left_eye_img, dtype="int32")
    img_arr_left = img_arr_left.reshape(42, 50, 1)
    right_eye_img.load()
    img_arr_right = np.asarray(right_eye_img, dtype="int32")
    img_arr_right = img_arr_right.reshape(42, 50, 1)

    left_eye_prediction = Left_Eye_Gaze_Model.predict(img_arr_left.reshape(1, 42, 50, 1))[0]
    right_eye_prediction = Right_Eye_Gaze_Model.predict(img_arr_right.reshape(1, 42, 50, 1))[0]
    pred_class = final_prediction(left_eye_prediction, right_eye_prediction)
    return class_name[pred_class]

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def recognize_speech_from_mic(recognizer, microphone):
    """Transcribe speech from recorded from `microphone`.

    Returns a dictionary with three keys:
    "success": a boolean indicating whether or not the API request was
               successful
    "error":   `None` if no error occured, otherwise a string containing
               an error message if the API could not be reached or
               speech was unrecognizable
    "transcription": `None` if speech could not be transcribed,
               otherwise a string containing the transcribed text
    """
    # check that recognizer and microphone arguments are appropriate type
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    # adjust the recognizer sensitivity to ambient noise and record audio
    # from the microphone
    with microphone as source:
        print("Listening......")
        recognizer.adjust_for_ambient_noise(source)
        # audio = recognizer.listen(source)
        audio=recognizer.record(source, duration = 2, offset = None)


    # set up the response object
    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    # try recognizing the speech in the recording
    # if a RequestError or UnknownValueError exception is caught,
    #     update the response object accordingly
    try:
        # print("Recognizing......")
        response["transcription"] = recognizer.recognize_google(audio,language='ar-AR') #,language='ar-AR'
        #response["transcription"] = recognizer.recognize_sphinx(audio)
    except sr.RequestError:
        # API was unreachable or unresponsive
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        # speech was unintelligible
        response["error"] = "Unable to recognize speech"

    return response


def Voice():
    recognizer = sr.Recognizer()
    # recognizer.energy_threshold = 1000
    # recognizer.non_speaking_duration = 0.2
    # recognizer.pause_threshold = 0.4
    microphone = sr.Microphone(device_index=1)
    guess = recognize_speech_from_mic(recognizer, microphone)
    what = guess["transcription"]

    print(what)
    if what is not None:
        if similar(what, "توقف") > similar(what, "تحرك"):
            return 0
        else:
            return 1
    else:
        return None
