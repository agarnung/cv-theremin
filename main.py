from theremin import Theremin 

def main():
    theremin = Theremin(min_frequency=200, max_frequency=600,
                        initial_frequency=440, initial_volume=0.5,
                        camera_id=0,
                        staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.9, minTrackCon=0.5
                        )

    theremin.start()

if __name__ == "__main__":
    main()