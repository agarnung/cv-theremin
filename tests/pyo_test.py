# Non-pytest test
# see https://github.com/belangeo/pyo
# see https://belangeo.github.io/pyo/about.html
# Using pyo to make it sound smoother and a slider to change frequency 

from pyo import *
import tkinter as tk

# Initial frequency and volume
current_frequency = 440.00 # manages fractional Hz
current_volume = 0.5       # initial volume (range from 0 to 1)

# Initial pyo configuration
server = Server().boot() # create a pyo server object and boot it

# Sine wave generator, let this sine stream to audio ouput
oscillator = Sine(freq=current_frequency, mul=current_volume).out()

server.start() # start the server

def update_frequency(value):
    """Updates the oscillator's frequency."""
    global current_frequency
    current_frequency = float(value)
    oscillator.freq = current_frequency

def update_volume(value):
    """Updates the oscillator's volume."""
    global current_volume
    current_volume = float(value) / 100 # convert to a range of [0, 1]
    oscillator.mul = current_volume     # adjust the volume

def start_gui():
    """Creates and runs the graphical interface."""
    def on_close():
        server.stop()
        server.shutdown()
        root.destroy()

    root = tk.Tk()
    root.title("Smooth Theremin")
    root.geometry("600x300")  # Increased height to accommodate both sliders

    # Slider to control frequency
    slider_frequency = tk.Scale(root, from_=200, to=800, resolution=0.01, orient=tk.HORIZONTAL, length=500, label="Frequency (Hz)")
    slider_frequency.set(current_frequency)
    slider_frequency.pack(pady=10)
    slider_frequency.config(command=update_frequency)

    # Slider to control volume
    slider_volume = tk.Scale(root, from_=0, to=100, resolution=1, orient=tk.HORIZONTAL, length=300, label="Volume (%)")
    slider_volume.set(current_volume * 100)  # Convert volume to percentage
    slider_volume.pack(pady=10)
    slider_volume.config(command=update_volume)

    # Exit button
    exit_button = tk.Button(root, text="Exit", command=on_close)
    exit_button.pack(pady=10)

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

start_gui()
