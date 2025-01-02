# Python interpreter used: ../bin/python3
# see https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html

import sys
import os
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/modules') # to include ../modules/CameraModule
from CameraModule import Camera

# Fixture to read the camera index from pytest options
@pytest.fixture
def camera_source(request):
    return request.config.getoption("--camera")

# Tests with support for the camera index
def test_camera_initialization(camera_source):
    """Test if the camera initializes correctly."""
    camera = Camera(camera_source)
    assert camera.cap.isOpened(), "Failed to open the camera"
    camera.release()

def test_camera_read(camera_source):
    """Test if frames can be read from the camera."""
    camera = Camera(camera_source)
    success, frame = camera.read()
    assert success, "Failed to read a frame from the camera"
    assert frame is not None, "The frame read is None"
    camera.release()

def test_camera_flip_horizontal(camera_source):
    """Test if flip_horizontal correctly flips the image."""
    camera = Camera(camera_source)
    success, frame = camera.read()
    assert success, "Failed to read a frame to test flip_horizontal"

    flipped_frame = camera.flip_horizontal(frame)
    assert flipped_frame is not None, "Flipped frame is None"
    assert flipped_frame.shape == frame.shape, "Flipped frame has a different shape"
    camera.release()

def test_camera_release(camera_source):
    """Test if the camera releases correctly."""
    camera = Camera(camera_source)
    camera.release()
    assert not camera.cap.isOpened(), "Failed to release the camera"