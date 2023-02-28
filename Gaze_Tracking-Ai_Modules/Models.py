def Get_Direction1(gaze):
    ret = None

    if gaze.is_blinking():
        ret = "Blinking"
    elif gaze.is_right():
        ret = "Looking right"
    elif gaze.is_left():
        ret = "Looking left"
    elif gaze.is_center():
        ret = "Looking center"

    return ret