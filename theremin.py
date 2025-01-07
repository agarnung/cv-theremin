from modules.AudioModule import Audio
from modules.CameraModule import Camera
from modules.HandTrackingModule import HandDetector

import cv2
import numpy as np

import skfuzzy as fuzz
from skfuzzy import control as ctrl

class Theremin:
    def __init__(self, min_frequency=200, max_frequency=600,
                 initial_frequency=440, initial_volume=0.0, 
                 camera_id=0,
                 staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5):
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.audio = Audio(initial_frequency=initial_frequency, initial_volume=initial_volume)
        self.camera = Camera(camera_id)
        self.hd = HandDetector(staticMode=staticMode, maxHands=maxHands, modelComplexity=modelComplexity, 
                               detectionCon=detectionCon, minTrackCon=minTrackCon)
        self.running = True # ensure it can start the loop
        self.initialize_production_rules()

    def calculate_fuzzy_sets(self, variable, min_val, max_val, use_gaussian):
        """
        Generates fuzzy sets (low, medium, high) for a variable using min and max values.
        """
        range_span = max_val - min_val
        if not use_gaussian:
            # Triangular membership seem to fail at boundaries
            low_max = min_val + range_span * 0.25
            medium_min = min_val + range_span * 0.125
            medium_max = min_val + range_span * 0.625
            high_min = min_val + range_span * 0.5

            variable['low'] = fuzz.trimf(variable.universe, [min_val, min_val, low_max])
            variable['medium'] = fuzz.trimf(variable.universe, [medium_min, (min_val + max_val) / 2, medium_max])
            variable['high'] = fuzz.trimf(variable.universe, [high_min, max_val, max_val])
        else:
            center_low = min_val + range_span * 0.125
            center_medium = (min_val + max_val) / 2
            center_high = max_val - range_span * 0.125
            sigma = range_span * 0.25  

            variable['low'] = fuzz.gaussmf(variable.universe, center_low, sigma)
            variable['medium'] = fuzz.gaussmf(variable.universe, center_medium, sigma)
            variable['high'] = fuzz.gaussmf(variable.universe, center_high, sigma)

    def initialize_production_rules(self):
        # Heuristic ranges
        self.min_openness = 20 
        self.max_openness = 100 
        self.min_proximity = 1 
        self.max_proximity = 25  
        self.min_distance = 30  
        self.max_distance = 250  
        openness_range = (self.min_openness, self.max_openness)    
        proximity_range = (self.min_proximity, self.max_proximity)     
        distance_range = (self.min_distance, self.max_distance)    
        frequency_range = (self.min_frequency, self.max_frequency) 

        # 1. Fuzzy variables definition
        #   Input variables
        openness = ctrl.Antecedent(np.arange(openness_range[0], openness_range[1] + 1, 1), 'openness')
        proximity = ctrl.Antecedent(np.arange(proximity_range[0], proximity_range[1] + 1, 1), 'proximity')
        distance = ctrl.Antecedent(np.arange(distance_range[0], distance_range[1] + 1, 1), 'distance')
        #   Output variable
        frequency = ctrl.Consequent(np.arange(frequency_range[0], frequency_range[1] + 1, 1), 'frequency')

        # 2. Create fuzzy sets
        self.calculate_fuzzy_sets(openness, *openness_range, use_gaussian=True)
        self.calculate_fuzzy_sets(proximity, *proximity_range, use_gaussian=True)
        self.calculate_fuzzy_sets(distance, *distance_range, use_gaussian=True)
        self.calculate_fuzzy_sets(frequency, *frequency_range, use_gaussian=True)

        # 3. Define fuzzy rules
        rule1 = ctrl.Rule(proximity['low'] & distance['high'] & openness['low'], frequency['low'])
        rule2 = ctrl.Rule(proximity['low'] & distance['high'] & openness['medium'], frequency['medium'])
        rule3 = ctrl.Rule(proximity['low'] & distance['high'] & openness['high'], frequency['high'])

        rule4 = ctrl.Rule(proximity['low'] & distance['medium'] & openness['low'], frequency['low'])
        rule5 = ctrl.Rule(proximity['low'] & distance['medium'] & openness['medium'], frequency['medium'])
        rule6 = ctrl.Rule(proximity['low'] & distance['medium'] & openness['high'], frequency['high'])

        rule7 = ctrl.Rule(proximity['low'] & distance['low'] & openness['low'], frequency['medium'])
        rule8 = ctrl.Rule(proximity['low'] & distance['low'] & openness['medium'], frequency['high'])
        rule9 = ctrl.Rule(proximity['low'] & distance['low'] & openness['high'], frequency['high'])

        rule10 = ctrl.Rule(proximity['medium'] & distance['high'] & openness['low'], frequency['medium'])
        rule11 = ctrl.Rule(proximity['medium'] & distance['high'] & openness['medium'], frequency['medium'])
        rule12 = ctrl.Rule(proximity['medium'] & distance['high'] & openness['high'], frequency['high'])

        rule13 = ctrl.Rule(proximity['medium'] & distance['medium'] & openness['low'], frequency['medium'])
        rule14 = ctrl.Rule(proximity['medium'] & distance['medium'] & openness['medium'], frequency['medium'])
        rule15 = ctrl.Rule(proximity['medium'] & distance['medium'] & openness['high'], frequency['high'])

        rule16 = ctrl.Rule(proximity['medium'] & distance['low'] & openness['low'], frequency['high'])
        rule17 = ctrl.Rule(proximity['medium'] & distance['low'] & openness['medium'], frequency['high'])
        rule18 = ctrl.Rule(proximity['medium'] & distance['low'] & openness['high'], frequency['high'])

        rule19 = ctrl.Rule(proximity['high'] & distance['high'] & openness['low'], frequency['low'])
        rule20 = ctrl.Rule(proximity['high'] & distance['high'] & openness['medium'], frequency['medium'])
        rule21 = ctrl.Rule(proximity['high'] & distance['high'] & openness['high'], frequency['medium'])

        rule22 = ctrl.Rule(proximity['high'] & distance['medium'] & openness['low'], frequency['low'])
        rule23 = ctrl.Rule(proximity['high'] & distance['medium'] & openness['medium'], frequency['medium'])
        rule24 = ctrl.Rule(proximity['high'] & distance['medium'] & openness['high'], frequency['medium'])

        rule25 = ctrl.Rule(proximity['high'] & distance['low'] & openness['low'], frequency['low'])
        rule26 = ctrl.Rule(proximity['high'] & distance['low'] & openness['medium'], frequency['medium'])
        rule27 = ctrl.Rule(proximity['high'] & distance['low'] & openness['high'], frequency['medium'])

        # 4. Create the control system
        frequency_ctrl = ctrl.ControlSystem([
            rule1, rule2, rule3, rule4, rule5, rule6,
            rule7, rule8, rule9, rule10, rule11, rule12,
            rule13, rule14, rule15, rule16, rule17, rule18,
            rule19, rule20, rule21, rule22, rule23, rule24,
            rule25, rule26, rule27
        ])
        self.frequency_simulator = ctrl.ControlSystemSimulation(frequency_ctrl)

    def compute_tone_crisp(self, width, height, right_hand) -> float:
        # Right hand controls frequency
        area = right_hand["bbox"][2] * right_hand["bbox"][3] # bounding box area

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
        
        print(f"Area: {area:.2f}, Geom Mean: {geom_mean:.2f}, Frequency: {new_frequency:.2f} Hz", end=" ")

        return new_frequency

    def compute_tone_fuzzy(self, width, height, right_hand) -> float:
        # Compute distance
        distance = right_hand['center'][0]

        # Compute proximity
        proximity = right_hand["bbox"][2] * right_hand["bbox"][3] # Bounding box area
        proximity = 100 * proximity / (width * height)

        # Compute openness 
        lmList2 = right_hand["lmList"]
        openness = 0
        if len(lmList2) > 0:
            points = [lmList2[i][0:2] for i in [4, 8, 12, 16, 20]]
            distances = []
            for i in range(len(points) - 1):
                dist = np.sqrt((points[i + 1][0] - points[i][0])**2 + (points[i + 1][1] - points[i][1])**2)
                distances.append(dist)
            if distances:
                openness = np.exp(np.mean(np.log(distances))) 

        proximity = int(np.clip(proximity, self.min_proximity, self.max_proximity))
        distance = int(np.clip(distance, self.min_distance, self.max_distance))
        openness = int(np.clip(openness, self.min_openness, self.max_openness))

        print(f"Proximity: {proximity}", end=" ")
        print(f"Distance: {distance}", end=" ")
        print(f"Openness: {openness}", end=" ")

        # Assign values to the antecedents (input variables)
        self.frequency_simulator.input['openness'] = openness
        self.frequency_simulator.input['proximity'] = proximity
        self.frequency_simulator.input['distance'] = distance

        # Perform fuzzy inference (computing the output)
        self.frequency_simulator.compute()
        if 'frequency' in self.frequency_simulator.output:
            new_frequency = self.frequency_simulator.output['frequency']
        else:
            print("ERROR loading output from ControlSystem")
            new_frequency = 0
        return new_frequency

    def compute_volume(self, height, left_hand) -> float:
        center2y = height - left_hand['center'][1]
        new_volume = min(max(center2y / height, 0), 1) # map Y to volume range [0, 1]
        return new_volume

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
                # frame = cv2.bilateralFilter(frame, 15, 75, 75) 

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
                        # new_frequency = self.compute_tone_crisp(width, height, right_hand)
                        new_frequency = self.compute_tone_fuzzy(width, height, right_hand)
                        self.audio.update_frequency(new_frequency)
                        print(f"Frequency: {new_frequency:.2f}", end=" ")

                    # Volume for left hand
                    if left_hand:
                        new_volume = self.compute_volume(height, left_hand)
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
