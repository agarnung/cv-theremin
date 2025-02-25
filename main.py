from theremin import Theremin 

def main():
    theremin = Theremin(use_fuzzy=False,
                        use_depth=True,
                        min_frequency=200, max_frequency=600,
                        initial_frequency=440, initial_volume=0.0,
                        camera_id=2,
                        staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5
                        )
    theremin.start()

if __name__ == "__main__":
    main()