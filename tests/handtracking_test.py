# Non-pytest test
# see https://github.com/cvzone/cvzone/blob/master/cvzone/HandTrackingModule.py

import sys
import os
import pytest
import cv2 as cv
import time
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/modules') # to include ../modules/HandTrackingModule
from HandTrackingModule import HandDetector

start_time = 0
fps = 0

random.seed()

def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

    while True:
        start = time.time()

        # Capture frame from camera
        success, img = cap.read()

        # Detect hand and draw it
        hands, img = detector.findHands(img, draw=True, flipType=True)

        # If any hands were detected, make a tone depending on the number made
        if hands:
            hand1 = hands[0]          # get the first hand detected
            lmList1 = hand1["lmList"] # list of 21 landmarks for the first hand
            bbox1 = hand1["bbox"]     # bounding box around the first hand (x,y,w,h coordinates)
            center1 = hand1['center'] # ceter coordinates of the first hand
            handType1 = hand1["type"] # type of the first hand ("Left" or "Right")

            fingers1 = detector.fingersUp(hand1)        # count the number of fingers up for the first hand
            print(f'H1 = {fingers1.count(1)}', end=" ") # print the count of fingers that are up

            print(" ")  # new line for better readability of the printed output

        # Check if a second hand is detected
        if len(hands) == 2:
            # Information for the second hand
            hand2 = hands[1]
            lmList2 = hand2["lmList"]
            bbox2 = hand2["bbox"]
            center2 = hand2['center']
            handType2 = hand2["type"]

            # Count the number of fingers up for the second hand
            fingers2 = detector.fingersUp(hand2)
            print(f'H2 = {fingers2.count(1)}', end=" ")

            # Calculate distance between the index fingers of both hands and draw it on the image
            length, info, img = detector.findDistance(lmList1[8][0:2], lmList2[8][0:2], img, color=(255, 0, 0),
                                                        scale=10)

        if (random.random() < .1):
            global fps 
            fps = round(1 / (time.time() - start), 2)
        
        img = cv.putText(img, str(fps), (25, 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv.LINE_AA)

        cv.imshow("Image", img)

        if cv.waitKey(1) == ord('q'):
            break
 
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()