"""
Audio Module
see https://github.com/belangeo/pyo
see https://belangeo.github.io/pyo/about.html
By: agarnung
"""

from pyo import *

class Audio:
    """
    Class representing an audio proxy for audio signals managment
    """

    def __init__(self, initial_frequency=440, initial_volume = 0.5) -> None:
        """
        Initialize the audio.

        :param source: Index of the camera device (default is 0) or a path to a local video file.
        """
        self.frequency = initial_frequency
        self.volume = initial_volume

        self.server = Server() # create the pyo server object but don't boot it yet

        self.server.boot() # boot the pyo server

        # Initialize oscillator with no server yet (it will be done later)
        self.oscillator = Sine(freq=initial_frequency, mul=initial_volume)

    def start(self):
        self.oscillator.out() # Start sending the signal to the output
        self.server.start()   # Start the server to process the audio

    def stop(self):
        """Stop the server gracefully."""
        try:
            self.server.stop()     # stop audio processing
            self.server.shutdown() # shutdown the server to release resources
        except Exception as e:
            print(f"Error while stopping the server: {e}")

    def showGUI(self):
        """Show the server's graphical interface."""
        self.server.gui(locals()) # show GUI and wait for user input

    def update_frequency(self, value):
        """Updates the oscillator's frequency."""
        self.frequency = float(value)
        self.oscillator.freq = self.frequency

    def update_volume(self, value):
        """Updates the oscillator's volume."""
        self.volume = float(value) / 100  # convert to a range of [0, 1]
        self.oscillator.mul = self.volume # adjust the volume
        
def main():
    audio = Audio(initial_frequency=440, initial_volume=0.5)
    audio.start()
    audio.showGUI()
    audio.stop()

if __name__ == "__main__":
    main()
