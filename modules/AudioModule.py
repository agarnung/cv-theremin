"""
Audio Module
see https://github.com/belangeo/pyo
see https://belangeo.github.io/pyo/about.html
By: agarnung
"""

from pyo import *


class AudioMeter:
    """Receives the RMS level of each output channel from pyo."""

    def __init__(self):
        self.rms = [0.0, 0.0]

    def setRms(self, left, right):
        self.rms = [float(left), float(right)]


class Audio:
    """
    Class representing an audio proxy for audio signals managment
    """

    def __init__(
        self,
        initial_frequency=440,
        initial_volume=0.5,
        output_device=None,
        sample_rate=48000,
        channels=2,
        buffer_size=256,
        duplex=0,
    ) -> None:
        self.frequency = initial_frequency
        self.volume = initial_volume
        self.channels = int(channels)

        self.server = Server(
            sr=int(sample_rate),
            nchnls=self.channels,
            buffersize=int(buffer_size),
            duplex=int(duplex),
        )
        device = self._resolve_output_device(output_device)
        if device is not None:
            self.server.setOutputDevice(device)
        self.output_device = device
        self.server.boot()
        if not self.server.getIsBooted():
            raise RuntimeError(
                "Pyo could not boot the audio server. Set audio.output_device "
                "in config.yaml to a valid output from `python3 -c "
                "'import pyo; pyo.pa_list_devices()'`."
            )

        self.meter = AudioMeter()
        self.server.setMeter(self.meter)
        self.oscillator = Sine(freq=initial_frequency, mul=1.0)
        self.volume_signal = Sig(initial_volume)
        self.output = self.oscillator * self.volume_signal

    def _resolve_output_device(self, device):
        if isinstance(device, str):
            names, indices = pa_get_output_devices()
            for index, name in zip(indices, names):
                if device.lower() in name.lower():
                    return int(index)
            raise ValueError(f"No audio output device matches {device!r}")

        if device is not None:
            index = int(device)
            if pa_get_output_max_channels(index) >= self.channels:
                return index
            print(
                f"Audio output device {index} has fewer than {self.channels} "
                "channels; selecting another output automatically."
            )

        return self._auto_output_device()

    def _auto_output_device(self):
        try:
            default = pa_get_default_output()
            if pa_get_output_max_channels(default) >= self.channels:
                return default
        except Exception:
            pass

        try:
            names, indices = pa_get_output_devices()
            for wanted in ("SPEAKER", "HEADPHONE", "JACK"):
                for index, name in zip(indices, names):
                    if (
                        wanted in name.upper()
                        and "HDMI" not in name.upper()
                        and pa_get_output_max_channels(index) >= self.channels
                    ):
                        return int(index)
            for index, name in zip(indices, names):
                if (
                    "HDMI" not in name.upper()
                    and pa_get_output_max_channels(index) >= self.channels
                ):
                    return int(index)
            for index in indices:
                if pa_get_output_max_channels(index) >= self.channels:
                    return int(index)
        except Exception:
            pass

        return None

    def start(self):
        self.output.out()
        self.server.start()

    def stop(self):
        """Stop the server gracefully."""
        try:
            self.server.stop()
            self.server.shutdown()
        except Exception as e:
            print(f"Error while stopping the server: {e}")

    def showGUI(self):
        """Show the server's graphical interface."""
        self.server.gui(locals())

    def update_frequency(self, value):
        """Updates the oscillator's frequency."""
        self.frequency = float(value)
        self.oscillator.freq = self.frequency

    def update_volume(self, value):
        """Updates the oscillator's volume."""
        self.volume = float(value) / 100
        self.volume_signal.value = self.volume


def main():
    audio = Audio(initial_frequency=440, initial_volume=0.5)
    audio.start()
    audio.showGUI()
    audio.stop()


if __name__ == "__main__":
    main()
