from modules.AudioModule import Audio
from modules.CameraModule import Camera
from modules.HandTrackingModule import HandDetector

import cv2
import numpy as np

class Theremin:
    def __init__(self, min_frequency=200, max_frequency=600,
                 initial_frequency=440, initial_volume=0.5, 
                 camera_id=0,
                 staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5):
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.audio = Audio(initial_frequency=initial_frequency, initial_volume=initial_volume)
        self.camera = Camera(camera_id)
        self.hd = HandDetector(staticMode=staticMode, maxHands=maxHands, modelComplexity=modelComplexity, 
                               detectionCon=detectionCon, minTrackCon=minTrackCon)
        self.running = True # ensure it can start the loop

    def start(self):
        self.audio.start()
        self.running = True

        try:
            while self.running:
                success, frame = self.camera.read()
                if not success:
                    print("Cannot read frame from camera.")
                    break

                frame = cv2.medianBlur(frame, 5)

                # Flip the frame and process for hand detection
                hands, frame = self.hd.findHands(frame)

                height, width = frame.shape[:2]

                if hands:
                    right_hand = None
                    left_hand = None

                    # Identify right and left hands based on the "type" field
                    for hand in hands:
                        if hand["type"] == "Right":
                            right_hand = hand
                        elif hand["type"] == "Left":
                            left_hand = hand
                    
                    # Frequency for right hand
                    if right_hand:
                        # Right hand controls frequency
                        area = right_hand["bbox"][2] * right_hand["bbox"][3]  # Bounding box area

                        # Compute hand "openess" or spread using distances between key points
                        lmList2 = right_hand["lmList"]
                        geom_mean = 0
                        if len(lmList2) > 0:
                            points = [lmList2[i][0:2] for i in [4, 8, 12, 16, 20]]
                            distances = []
                            for i in range(len(points) - 1):
                                dist = np.sqrt((points[i + 1][0] - points[i][0])**2 + (points[i + 1][1] - points[i][1])**2)
                                distances.append(dist)
                            if distances:
                                geom_mean = np.exp(np.mean(np.log(distances))) 

                        # Normalize the area and geom_mean to a range that fits the theremin frequency (100-800 Hz)
                        max_area = 0.25 * height * width          # heuristic maximum possible area for normalization
                        area_normalized = min(area / max_area, 1) # normalize to [0, 1]
                        frequency_from_area = self.min_frequency + area_normalized * self.max_frequency # map area to the frequency range
                        
                        # Normalizing hand spread (geom_mean) to influence frequency
                        max_geom_mean = 150                                      # heuristic max value for hand spread
                        geom_mean_normalized = min(geom_mean / max_geom_mean, 1) # normalize to [0, 1]
                        frequency_from_geom = self.min_frequency + geom_mean_normalized * self.max_frequency  # map spread to the frequency range

                        # Combine both factors for final frequency
                        new_frequency = 0.75 * frequency_from_area + 0.25 * frequency_from_geom 
                        self.audio.update_frequency(new_frequency)
                        
                        print(f"Area: {area}, Geom Mean: {geom_mean}, Frequency: {new_frequency:.2f} Hz", end=" ")

                    # Volume for left hand
                    if left_hand:
                        center2y = height - left_hand['center'][1]
                        new_volume = min(max(center2y / height, 0), 1) # map Y to volume range [0, 1]
                        self.audio.update_volume(new_volume * 100)
                        print(f"Volume: {new_volume * 100:.2f}", end=" ")
                    else:
                        self.audio.update_volume(0) # no left hand detected, mute volume
                        print("Volume: 0", end=" ")

                    print() 

                else:
                    self.audio.update_frequency(0) # no hands detected, mute frequency
                    self.audio.update_volume(0)    # mute volume as well
                    print("No hands detected, Frequency: 0, Volume: 0")

                cv2.imshow("Theremin View", frame)

                # Exit the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.stop()

    def stop(self):
        self.running = False
        self.audio.stop()
        self.camera.release()
        cv2.destroyAllWindows()
