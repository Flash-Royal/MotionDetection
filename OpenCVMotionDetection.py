import cv2
import numpy as np
import imutils

def createDataSet(filename):
    i = 0
    try:
        f = open("train/motionRes.txt", 'r')
        for line in f:
            i += 1
        f = open("train/motionRes.txt", 'a')
    except:
        f = open("train/motionRes.txt", 'w')
    cam = cv2.VideoCapture(filename)
    firstFrame = None

    while True:
        reg, frame = cam.read()
        isMotion = 0
        text = "Unoccupied"

        frame = imutils.resize(frame, width = 500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (17, 17), 0)
        newFrame = frame.copy()

        if firstFrame is None:
            firstFrame = frame
            continue

        firstFrameGray = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
        firstFrameGray = cv2.GaussianBlur(firstFrameGray, (17, 17), 0)
        frameDelta = cv2.absdiff(firstFrameGray, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.dilate(thresh, None, iterations = 2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            if cv2.contourArea(c) < 30:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(newFrame,(x,y),(x+w,y+h),(0,255,0),1)
            text = "Occupied"
            isMotion = 1

        #cv2.putText(frame, f"Status: {text}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Video", frame)
        cv2.imshow("Security Feed", newFrame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)

        cv2.imwrite(f"train/curImages/image_{i}.jpg", frame)
        cv2.imwrite(f"train/prevImages/image_{i}.jpg", firstFrame)
        f.write(f"{isMotion}\n")
        firstFrame = frame
        i += 1
        key = cv2.waitKey(60)
        if key == 27:
            break

    cam.release()
    f.close()
    cv2.destroyAllWindows()

createDataSet("cars3.webm")
