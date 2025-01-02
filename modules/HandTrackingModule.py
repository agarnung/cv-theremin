"""
Hand Tracking Module
Based on https://github.com/cvzone/cvzone/blob/master/cvzone/HandTrackingModule.py
"""

import math

import cv2
import mediapipe as mp
import random
import time 

class HandDetector:
    """
    Detects hands using the mediapipe library. Exports the landmarks in pixel format.
    Adds additional functionalities like counting how many fingers are up or the distance
    between two fingers. Also provides bounding box information for the detected hand.
    """

    def __init__(self, staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5):
        """
        :param staticMode: In static mode, detection is done on each image individually, which is slower.
        :param maxHands: Maximum number of hands to detect.
        :param modelComplexity: Complexity of the hand landmark model: 0 or 1.
        :param detectionCon: Minimum detection confidence threshold.
        :param minTrackCon: Minimum tracking confidence threshold.
        """
        self.staticMode = staticMode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.staticMode,
                                        max_num_hands=self.maxHands,
                                        model_complexity=modelComplexity,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)

        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]  # tip identifiers for the fingers
        self.fingers = []  # list to store the finger states (up or down)
        self.lmList = []  # list to store the hand landmark coordinates

    def findHands(self, img, draw=True, flipType=True):
        """
        Detects hands in a BGR image.
        :param img: Image in which to detect hands.
        :param draw: Flag to draw landmarks and hand outline on the image.
        :return: Detected hands and the processed image.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert image to RGB for mediapipe
        self.results = self.hands.process(imgRGB)
        allHands = []  # list to store all detected hands
        h, w, c = img.shape  # get image dimensions

        if self.results.multi_hand_landmarks:  # if hands are detected
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}  # dictionary to store information about the hand

                # lmList: list of 21 landmarks for the hand
                mylmList = []  # list to store the landmarks
                xList = []  # list for x-coordinates
                yList = []  # list for y-coordinates
                for id, lm in enumerate(handLms.landmark):  # enumerate through each landmark
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)  # calculate pixel positions
                    mylmList.append([px, py, pz])  # add to landmarks list
                    xList.append(px)  # add x-coordinate to xList
                    yList.append(py)  # add y-coordinate to yList

                # bbox: Bounding box around the hand
                xmin, xmax = min(xList), max(xList)  # min and max x coordinates
                ymin, ymax = min(yList), max(yList)  # min and max y coordinates
                boxW, boxH = xmax - xmin, ymax - ymin  # width and height of the bounding box
                bbox = xmin, ymin, boxW, boxH  # bounding box coordinates
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)  # center of the bounding box

                myHand["lmList"] = mylmList  # store landmarks in dictionary
                myHand["bbox"] = bbox  # store bounding box
                myHand["center"] = (cx, cy)  # store center coordinates

                if flipType:  # if flipType is True, adjust hand type
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"  # flip to "Left" if right hand
                    else:
                        myHand["type"] = "Right"  # flip to "Right" if left hand
                else:
                    myHand["type"] = handType.classification[0].label  # set hand type as per the mediapipe result
                allHands.append(myHand)  # add hand to allHands list

                # draw landmarks and bounding box on image if draw is True
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)  # draw hand landmarks
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)  # draw bounding box around hand
                    cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)  # label hand type

        return allHands, img  # return detected hands and image with markings

    def fingersUp(self, myHand):
        """
        Detects how many fingers are up and returns them in a list.
        :return: List of fingers that are up (1 for up, 0 for down).
        """
        fingers = []  # list to store which fingers are up
        myHandType = myHand["type"]  # get hand type (left or right)
        myLmList = myHand["lmList"]  # get the landmarks list

        # Thumb
        if myHandType == "Right":
            if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                fingers.append(1)  # thumb is up if x-coordinate of tip is greater
            else:
                fingers.append(0)  # thumb is down if not
        else:
            if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                fingers.append(1)  # thumb is up if x-coordinate of tip is less
            else:
                fingers.append(0)  # thumb is down if not

        # 4 Fingers
        for id in range(1, 5):  # loop through the remaining fingers
            if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                fingers.append(1)  # finger is up if y-coordinate of tip is smaller
            else:
                fingers.append(0)  # finger is down if not
        return fingers

    def findDistance(self, p1, p2, img=None, color=(255, 0, 255), scale=5):
        """
        Finds the distance between two landmarks and optionally draws it on the image.
        :param p1: point 1 (x1, y1)
        :param p2: point 2 (x2, y2)
        :param img: image to draw the output on (optional)
        :return: Distance between the points, image with drawing (if any), and line information
        """

        x1, y1 = p1  # extract coordinates of point 1
        x2, y2 = p2  # extract coordinates of point 2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # calculate the center between the two points
        length = math.hypot(x2 - x1, y2 - y1)  # calculate the Euclidean distance
        info = (x1, y1, x2, y2, cx, cy)  # store line and center info

        if img is not None:  # if an image is provided, draw the line and circles
            cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)  # draw circle at point 1
            cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)  # draw circle at point 2
            cv2.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))  # draw line between points
            cv2.circle(img, (cx, cy), scale, color, cv2.FILLED)  # draw circle at the center

        return length, info, img  # return distance, line information, and updated image

# Principal program for testing the module
def main():
    start_time = 0
    fps = 0

    cap = cv2.VideoCapture(0)
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
            fps = round(1 / (time.time() - start), 2)
        
        img = cv2.putText(img, str(fps), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) == ord('q'):
            break
 
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
