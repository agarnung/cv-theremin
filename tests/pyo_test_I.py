# see https://github.com/belangeo/pyo
# see https://belangeo.github.io/pyo/about.html

import sys
import os
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/modules') # to include ../modules/AudioModule
from AudioModule import Audio  # Replace with the correct module name/path

# Fixture to initialize the Audio object
@pytest.fixture
def audio():
    audio_instance = Audio(initial_frequency=440, initial_volume=0.5)
    yield audio_instance
    audio_instance.stop()

def test_audio_initialization(audio):
    """Test if the audio object initializes correctly."""
    assert audio.server.getIsBooted(), "Audio server did not boot correctly"
    assert audio.oscillator.freq == 440, "Initial frequency is incorrect"
    assert audio.oscillator.mul == 0.5, "Initial volume is incorrect"

def test_audio_start(audio):
    """Test if the audio server starts correctly."""
    audio.start()
    assert audio.server.getIsStarted(), "Audio server did not start correctly"

def test_audio_stop(audio):
    """Test if the audio server stops correctly."""
    audio.start()
    audio.stop()
    assert not audio.server.getIsStarted(), "Audio server did not stop correctly"

def test_update_frequency(audio):
    """Test if the frequency update works as expected."""
    new_frequency = 880
    audio.update_frequency(new_frequency)
    assert audio.oscillator.freq == new_frequency, "Frequency update failed"

def test_update_volume(audio):
    """Test if the volume update works as expected."""
    new_volume = 75  # Percentage
    audio.update_volume(new_volume)
    assert audio.oscillator.mul == new_volume / 100, "Volume update failed"

def test_audio_gui(audio):
    """Test if the GUI method does not crash."""
    try:
        audio.showGUI()
        audio.stop()
    except Exception as e:
        pytest.fail(f"Audio GUI method raised an exception: {e}")