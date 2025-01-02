"""
Camera Module
Based on https://github.com/AnodeGrindYo/AI-Theremin
"""

import cv2
from typing import Tuple, Union

class Camera:
    """
    Class representing a camera to capture frames.
    """

    def __init__(self, source: Union[int, str] = 0) -> None:
        """
        Initialize the camera.

        :param source: Index of the camera device (default is 0) or a path to a local video file.
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera or video source: {source}")

    def read(self) -> Tuple[bool, Union[None, cv2.Mat]]:
        """
        Read a frame from the camera.

        :return: A tuple (success, frame), where success is a boolean and frame is the captured image or None.
        """
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame

    def flip_horizontal(self, image: cv2.Mat) -> cv2.Mat:
        """
        Flip the image horizontally.

        :param image: The input image.
        :return: The horizontally flipped image.
        """
        return cv2.flip(image, 1)

    def release(self) -> None:
        """
        Release the camera resource.
        """
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    """
    Main function to demonstrate the usage of the Camera class.
    """
    source = 0              # Change to a file path to use a video file instead of the camera
    camera = Camera(source) # Initialize camera or video source

    print("Press 'q' to quit.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Flip the frame horizontally and convert to grayscale
            flipped_frame = camera.flip_horizontal(frame)
            gray_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2GRAY)

            # Display the processed frame
            cv2.imshow('Camera - Grayscale', gray_frame)

            # Exit loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release the camera or close video source and close any OpenCV windows
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
