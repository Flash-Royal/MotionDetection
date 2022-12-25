import cv2
import numpy as np
import imutils
import tensorflow as tf


cam = cv2.VideoCapture('cars4.webm')
firstFrame = None
model = tf.keras.models.load_model('any3.h5')
while True:
    reg, frame = cam.read()
    isMotion = 0
    text = ""

    frame = imutils.resize(frame, width = 500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (17, 17), 0)
    newFrame = frame.copy()

    if firstFrame is None:
        firstFrame = frame
        continue

    scale_percent = 20 # percent of original size
    # (56, 100, 3) - 20% (281, 500, 3) - 100%
    width = int(firstFrame.shape[1] * scale_percent / 100)
    height = int(firstFrame.shape[0] * scale_percent / 100)
    dim = (width, height)

    firstFrame = cv2.resize(firstFrame, dim, interpolation = cv2.INTER_AREA)
    newFrame = cv2.resize(newFrame, dim, interpolation = cv2.INTER_AREA)


    prediction = model.predict([np.array([firstFrame]), np.array([newFrame])])
    print(prediction)
    if prediction[0][1] > prediction[0][0]:
        text = "Have Motion"
    else:
        text = "Haven't Motion"

    cv2.putText(frame, f"Status: {text}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Video", frame)

    firstFrame = frame

    key = cv2.waitKey(60)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
