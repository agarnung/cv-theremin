import copy
import time

import cv2
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from modules.AudioModule import Audio
from modules.CameraModule import Camera
from modules.ConfigModule import Config
from modules.DepthAnythingModule import DepthAnythingTheremin
from modules.DepthThereminModule import DepthTheremin
from modules.HandTrackingModule import HandDetector
from modules.HudModule import Hud
from modules.OneEuroFilter import OneEuroFilter

MODES = ("naive", "fuzzy", "pinhole", "depth_anything")


class Theremin:
    def __init__(self, config_path="config.yaml"):
        self.config = Config(config_path)
        self._config_data = {}
        self.audio = None
        self.camera = None
        self.hd = None
        self.depth_module = None
        self.frequency_filter = None
        self.volume_filter = None
        self.use_filter = False
        self.use_fuzzy = False
        self.use_depth = False
        self.depth_backend = None
        self._reload_enabled = True
        self._reload_interval = 1.0
        self._last_config_check = 0.0
        self.running = True
        self._last_frame_time = time.perf_counter()
        self._apply_config(self.config.data, initial=True)

    @staticmethod
    def _section(cfg, key):
        value = cfg.get(key)
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _camera_kwargs(cfg):
        camera = Theremin._section(cfg, "camera")
        return {
            "source": camera.get("index", 0),
            "width": camera.get("width"),
            "height": camera.get("height"),
            "fps_limit": camera.get("fps_limit"),
        }

    @staticmethod
    def _hand_kwargs(cfg):
        hand = Theremin._section(cfg, "hand_tracking")
        return {
            "staticMode": hand.get("static_mode", False),
            "maxHands": hand.get("max_hands", 2),
            "modelComplexity": hand.get("model_complexity", 1),
            "detectionCon": hand.get("detection_confidence", 0.5),
            "minTrackCon": hand.get("tracking_confidence", 0.5),
        }

    @staticmethod
    def _filter_kwargs(cfg):
        filt = Theremin._section(cfg, "filter")
        return {
            "frequency_min_cutoff": filt.get("frequency_min_cutoff", 1.0),
            "frequency_beta": filt.get("frequency_beta", 0.05),
            "volume_min_cutoff": filt.get("volume_min_cutoff", 1.0),
            "volume_beta": filt.get("volume_beta", 0.05),
            "d_cutoff": filt.get("d_cutoff", 1.0),
        }

    def _make_camera(self, cfg):
        return Camera(**self._camera_kwargs(cfg))

    def _make_hand_detector(self, cfg):
        return HandDetector(**self._hand_kwargs(cfg))

    def _make_depth_module(self, mode, cfg, min_frequency, max_frequency):
        depth = Theremin._section(cfg, "depth")
        if mode == "depth_anything":
            return DepthAnythingTheremin(
                min_frequency=min_frequency,
                max_frequency=max_frequency,
                model=depth.get("model", "LiheYoung/depth-anything-small-hf"),
                input_size=depth.get("input_size", 336),
                device=depth.get("device", "auto"),
                half_precision=depth.get("half_precision", True),
                inference_ttl=depth.get("inference_ttl", 2.0),
                patch_scale=depth.get("patch_scale", 0.25),
            )
        return DepthTheremin(
            min_frequency=min_frequency,
            max_frequency=max_frequency,
        )

    def _make_filters(self, cfg):
        filt = self._filter_kwargs(cfg)
        return (
            OneEuroFilter(
                min_cutoff=filt["frequency_min_cutoff"],
                beta=filt["frequency_beta"],
                d_cutoff=filt["d_cutoff"],
            ),
            OneEuroFilter(
                min_cutoff=filt["volume_min_cutoff"],
                beta=filt["volume_beta"],
                d_cutoff=filt["d_cutoff"],
            ),
        )

    def _apply_config(self, cfg, initial=False):
        old = self._config_data
        pitch = Theremin._section(cfg, "pitch")
        mode = pitch.get("mode", "depth_anything")
        if mode not in MODES:
            raise ValueError(f"Unknown pitch.mode {mode!r}; use one of {MODES}.")

        min_frequency = float(pitch.get("min_frequency", 200))
        max_frequency = float(pitch.get("max_frequency", 600))
        volume = Theremin._section(cfg, "volume")
        volume_source = str(volume.get("source", "left"))
        if volume_source not in ("both", "auto", "left", "right"):
            raise ValueError(
                f"Unknown volume.source {volume_source!r}; "
                "use both, auto, left or right."
            )
        volume_fallback = float(volume.get("fallback", 0.0))
        volume_min = float(volume.get("min", 0.0))
        volume_max = float(volume.get("max", 1.0))
        volume_gamma = float(volume.get("gamma", 1.0))
        camera = Theremin._section(cfg, "camera")
        mirror = bool(camera.get("mirror", False))
        hand = Theremin._section(cfg, "hand_tracking")
        flip_type = bool(hand.get("flip_type", False))
        use_fuzzy = mode == "fuzzy"
        use_depth = mode in ("pinhole", "depth_anything")
        depth_backend = mode if use_depth else None

        display = Theremin._section(cfg, "display")
        display_mode = str(display.get("mode", "hud")).lower()
        if display_mode not in ("hud", "raw"):
            raise ValueError(
                f"Unknown display.mode {display_mode!r}; use hud or raw."
            )
        draw_overlay = display_mode == "hud"

        reload = Theremin._section(cfg, "reload")
        reload_enabled = bool(reload.get("enabled", True))
        reload_interval = float(reload.get("check_interval", 1.0))

        if initial:
            audio = Theremin._section(cfg, "audio")
            self.audio = Audio(
                initial_frequency=audio.get("initial_frequency", 440),
                initial_volume=audio.get("initial_volume", 0.0),
                output_device=audio.get("output_device"),
                sample_rate=audio.get("sample_rate", 48000),
                channels=audio.get("channels", 2),
                buffer_size=audio.get("buffer_size", 256),
                duplex=audio.get("duplex", 0),
            )
            self.camera = self._make_camera(cfg)
            self.hd = self._make_hand_detector(cfg)
        else:
            if Theremin._section(cfg, "camera") != Theremin._section(old, "camera"):
                camera = self._make_camera(cfg)
                self.camera.release()
                self.camera = camera
            if Theremin._section(cfg, "hand_tracking") != Theremin._section(old, "hand_tracking"):
                detector = self._make_hand_detector(cfg)
                if hasattr(self.hd, "close"):
                    self.hd.close()
                self.hd = detector

        self.min_frequency = min_frequency
        self.max_frequency = max_frequency

        old_pitch = Theremin._section(old, "pitch")
        old_mode = old_pitch.get("mode")
        old_min_frequency = old_pitch.get("min_frequency")
        old_max_frequency = old_pitch.get("max_frequency")
        old_depth = Theremin._section(old, "depth")
        depth = Theremin._section(cfg, "depth")
        model_keys = ("model", "input_size", "device", "half_precision")
        model_changed = any(depth.get(k) != old_depth.get(k) for k in model_keys)
        backend_changed = initial or mode != old_mode

        if backend_changed or (use_depth and model_changed):
            depth_module = None
            if use_depth:
                depth_module = self._make_depth_module(
                    mode, cfg, min_frequency, max_frequency
                )
            previous = self.depth_module
            self.depth_module = depth_module
            if previous is not None and hasattr(previous, "stop"):
                previous.stop()
            if depth_module is not None and not initial and hasattr(depth_module, "start"):
                depth_module.start()
        elif self.depth_module is not None and hasattr(self.depth_module, "update_config"):
            self.depth_module.update_config(
                min_frequency=min_frequency,
                max_frequency=max_frequency,
                inference_ttl=depth.get("inference_ttl", 2.0),
                patch_scale=depth.get("patch_scale", 0.25),
            )

        if use_fuzzy and (
            initial
            or mode != old_mode
            or min_frequency != old_min_frequency
            or max_frequency != old_max_frequency
        ):
            self.initialize_production_rules()

        self.use_fuzzy = use_fuzzy
        self.use_depth = use_depth
        self.depth_backend = depth_backend
        self.flip_type = flip_type
        self.mirror = mirror
        self.volume_source = volume_source
        self.volume_fallback = volume_fallback
        self.volume_min = volume_min
        self.volume_max = volume_max
        self.volume_gamma = volume_gamma
        self.display_mode = display_mode
        self.draw_overlay = draw_overlay
        debug = Theremin._section(cfg, "debug")
        show_hud = bool(debug.get("show_hud", True))
        if initial or debug != Theremin._section(old, "debug"):
            self.show_hud = show_hud
            self.hud = Hud(
                svg_path=debug.get("hud_svg", "assets/theremin_pov.svg"),
                scale=debug.get("hud_scale", 0.75),
                opacity=debug.get("hud_opacity", 0.85),
                position=debug.get("hud_position", "bottom_center"),
            ) if show_hud else None

        filter_cfg = Theremin._section(cfg, "filter")
        if initial or filter_cfg != Theremin._section(old, "filter"):
            use_filter = bool(filter_cfg.get("enabled", True))
            if use_filter:
                frequency_filter, volume_filter = self._make_filters(cfg)
            else:
                frequency_filter = volume_filter = None
            self.use_filter = use_filter
            self.frequency_filter = frequency_filter
            self.volume_filter = volume_filter

        self._reload_enabled = reload_enabled
        self._reload_interval = reload_interval
        self._config_data = copy.deepcopy(cfg)

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

    def compute_volume(self, height, hand) -> float:
        lm_list = hand.get("lmList")
        y = lm_list[0][1] if lm_list else hand["center"][1]
        return min(max((height - y) / height, 0.0), 1.0)

    def _select_volume_hands(self, right_hand, left_hand):
        if self.volume_source == "left":
            return [left_hand] if left_hand is not None else []
        if self.volume_source == "right":
            return [right_hand] if right_hand is not None else []
        if self.volume_source == "auto":
            hand = left_hand if left_hand is not None else right_hand
            return [hand] if hand is not None else []
        return [hand for hand in (left_hand, right_hand) if hand is not None]

    def _draw_hud(self, frame, hands=None):
        if self.hud is None:
            return frame
        span = max(self.max_frequency - self.min_frequency, 1e-6)
        pitch = (self.audio.frequency - self.min_frequency) / span
        rms = max(self.audio.meter.rms)
        return self.hud.draw(
            frame,
            pitch=pitch,
            volume=self.audio.volume,
            frequency=self.audio.frequency,
            rms=rms,
            hands=hands,
        )

    def start(self):
        self.audio.start()
        if self.use_depth and hasattr(self.depth_module, "start"):
            self.depth_module.start()
        self.running = True
        self._last_frame_time = time.perf_counter()
        self._last_config_check = self._last_frame_time

        try:
            while self.running:
                now = time.perf_counter()
                dt = min(max(now - self._last_frame_time, 1e-3), 0.1)
                self._last_frame_time = now

                if self._reload_enabled and now - self._last_config_check >= self._reload_interval:
                    self._last_config_check = now
                    if self.config.reload():
                        try:
                            self._apply_config(self.config.data)
                            print("Config reloaded.")
                        except Exception as exc:
                            print(f"Config reload failed: {exc}")

                success, frame = self.camera.read()
                if not success:
                    print("Cannot read frame from camera.")
                    break

                if self.mirror:
                    frame = cv2.flip(frame, 1)

                if self.use_depth and hasattr(self.depth_module, "submit"):
                    self.depth_module.submit(frame)

                raw_frame = frame
                detection_frame = cv2.medianBlur(frame, 5)
                # frame = cv2.bilateralFilter(frame, 15, 75, 75)

                hands, detection_frame = self.hd.findHands(
                    detection_frame,
                    draw=self.draw_overlay,
                    flipType=self.flip_type,
                )
                frame = detection_frame if self.draw_overlay else raw_frame

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
                        if self.use_depth:
                            new_frequency, depth = self.depth_module.compute_tone_depth(
                                right_hand["bbox"], right_hand["center"]
                            )
                            if new_frequency is not None:
                                if self.use_filter:
                                    new_frequency = self.frequency_filter.filter(new_frequency, dt)
                                self.audio.update_frequency(new_frequency)
                                if self.depth_backend == "depth_anything":
                                    print(f"Depth value: {depth:.2f}, Frequency: {new_frequency:.2f} Hz", end=" ")
                                else:
                                    print(f"Depth: {depth:.2f} cm, Frequency: {new_frequency:.2f} Hz", end=" ")
                            else:
                                print("Depth: waiting, Frequency: unchanged", end=" ")
                        else:
                            if not self.use_fuzzy:
                                new_frequency = self.compute_tone_crisp(width, height, right_hand)
                            else:
                                new_frequency = self.compute_tone_fuzzy(width, height, right_hand)
                            if self.use_filter:
                                new_frequency = self.frequency_filter.filter(new_frequency, dt)
                            self.audio.update_frequency(new_frequency)
                            print(f"Frequency: {new_frequency:.2f}", end=" ")
                    else:
                        if self.use_filter:
                            self.frequency_filter.reset()

                    # Volume follows every visible hand by default
                    volume_hands = self._select_volume_hands(right_hand, left_hand)
                    if volume_hands:
                        values = [self.compute_volume(height, hand) for hand in volume_hands]
                        new_volume = sum(values) / len(values)
                        new_volume = self.volume_min + (
                            self.volume_max - self.volume_min
                        ) * (new_volume ** self.volume_gamma)
                        if self.use_filter:
                            new_volume = self.volume_filter.filter(new_volume, dt)
                        self.audio.update_volume(new_volume * 100)
                        print(f"Volume: {new_volume * 100:.2f}", end=" ")
                    else:
                        if self.use_filter:
                            self.volume_filter.reset()
                        fallback = self.volume_fallback if right_hand else 0.0
                        self.audio.update_volume(fallback * 100)
                        if right_hand:
                            print(f"Volume: {fallback * 100:.2f} (fallback)", end=" ")
                        else:
                            print("Volume: 0", end=" ")

                    print() 

                else:
                    if self.use_filter:
                        self.frequency_filter.reset()
                        self.volume_filter.reset()
                    self.audio.update_volume(0)    # mute volume as well
                    print("No hands detected, Volume: 0")

                if self.draw_overlay:
                    frame = self._draw_hud(frame, hands)
                cv2.imshow("Theremin View", frame)

                # Exit the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.stop()

    def stop(self):
        self.running = False
        if self.use_depth and hasattr(self.depth_module, "stop"):
            self.depth_module.stop()
        self.audio.stop()
        self.camera.release()
        cv2.destroyAllWindows()
