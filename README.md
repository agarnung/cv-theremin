# cv-theremin

A hand-tracking, camera-based digital theremin.

The right hand controls pitch and the left hand controls volume. If no hands are
detected, the volume is muted. By default only the left hand controls volume:
move it up or down to change it. Set `volume.source` to `right`, `auto` or
`both` if you want a different hand to control it.

## How it works

The theremin reads frames from a webcam, tracks the hands with MediaPipe and turns
the right hand's position or depth into a frequency. Pitch is computed with exactly
one of these modes:

| Mode | Pitch source |
|---|---|
| Naïve | 2D hand size and finger spread |
| Fuzzy logic | Hand pose and geometry through a fuzzy controller |
| Pinhole depth | Hand distance estimated from the bounding-box width and the camera calibration |
| Depth Anything | Relative monocular depth from a Depth Anything model |

`Depth Anything` is the default. It runs in a background thread, so hand tracking
and audio never wait for inference. The model returns a depth map, normalised per
frame to 0 (farthest) - 1 (nearest). The theremin samples the depth at the hand
and maps that value directly to the configured frequency range.

On top of any pitch mode, the **1€ filter** smooths frequency and volume. It is an
adaptive low-pass filter: it filters more when the hand is slow and reacts faster
when the hand moves quickly. This reduces jitter without adding noticeable lag
(Casiez et al., CHI 2012).

## Requirements

- Python 3.10 or 3.12 (MediaPipe 0.10.x does not provide `mediapipe.solutions` on Python 3.14)
- A webcam
- `opencv-contrib-python`
- `mediapipe`
- `pyo`
- `scikit-fuzzy`
- `PyTorch`
- `transformers`
- `Pillow`
- `PyYAML`

`numpy` is pinned below 2.0 because MediaPipe 0.10.x requires it.

Depth Anything is strongly recommended with a CUDA-capable GPU. On CPU it still
works, but inference is slow.

## Setup

Create and activate a virtual environment with Python 3.10 or 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

If you already created `.venv` with Python 3.14, remove it first:

```bash
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python3 -m pip install -r requirements.txt
```

If you have an NVIDIA GPU, install the CUDA build of PyTorch before the rest of
the requirements. Pick the index that matches your driver (for example `cu128`):

```bash
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Run

```bash
python3 main.py
```

`main.py` only loads `config.yaml`; all settings live there. The Depth Anything
model is downloaded from Hugging Face on the first run and then reused from the
local cache.

## Configuration

Edit `config.yaml` while the theremin is running. It is checked every
`reload.check_interval` seconds and changes are applied live:

```yaml
camera:
  index: 0                 # color camera; index 2 is often the IR camera
  width: 640
  height: 480
  fps_limit: 30
  mirror: true             # mirror the preview; if false, set flip_type to true

audio:
  output_device: null      # null = auto; or an index from pyo.pa_list_devices()
  sample_rate: 48000
  channels: 2
  buffer_size: 256
  duplex: 0
  initial_frequency: 440
  initial_volume: 0.0

pitch:
  mode: "depth_anything"   # naive | fuzzy | pinhole | depth_anything
  min_frequency: 200
  max_frequency: 600

volume:
  source: "left"            # left = only the left hand controls volume
  fallback: 0.5            # volume when the left hand is not visible
  min: 0.0                  # volume at the bottom of the frame
  max: 1.0                  # volume at the top of the frame
  gamma: 1.5                # higher = makes low volumes quieter; lower = makes low volumes louder

depth:
  model: "LiheYoung/depth-anything-small-hf"
  input_size: 336
  device: "auto"           # auto | cuda | mps | cpu
  half_precision: true
  inference_ttl: 2.0
  patch_scale: 0.25

filter:
  enabled: true
  frequency_min_cutoff: 1.0
  frequency_beta: 0.05
  volume_min_cutoff: 2.0
  volume_beta: 0.5
  d_cutoff: 1.0

display:
  mode: "hud"               # hud = theremin, hands and values; raw = camera frame only

hand_tracking:
  static_mode: false
  max_hands: 2
  model_complexity: 1
  detection_confidence: 0.5
  tracking_confidence: 0.5
  flip_type: false          # false with a mirrored preview; true with a raw camera frame

reload:
  enabled: true
  check_interval: 1.0

debug:
  show_hud: true            # draw the theremin HUD; false = hide it
  hud_svg: "assets/theremin_pov.svg"  # SVG theremin drawn in perspective
  hud_scale: 0.75           # lower = smaller HUD; higher = larger HUD
  hud_opacity: 0.85          # lower = more transparent; higher = more opaque
  hud_position: "bottom_center"  # bottom_center | bottom_left | bottom_right | center | top_center
```

What is applied live:

- `pitch.mode`, `pitch.min_frequency` and `pitch.max_frequency`.
- All `depth` tuning values. Changing `model`, `input_size`, `device` or
  `half_precision` reloads the depth model.
- All `filter` values.
- `camera`, `display` and `hand_tracking` settings.
- `reload.enabled` and `reload.check_interval`.
- All `debug` settings, including the SVG HUD.

`audio` settings are only read at startup. With `output_device: null`, pyo uses the
default output if it has enough channels; otherwise it picks the first non-HDMI
output with the requested channels. To list devices and choose one manually:

```bash
python3 -c "import pyo; pyo.pa_list_devices()"
```

ALSA `pcm_dsnoop`/`pipewire` warnings on startup are normal on PipeWire systems.
On Fedora, installing `pipewire-alsa` removes most of them.

If the preview is grayscale and flickers, you are probably reading the IR camera
(usually index `2` on laptops with Windows Hello). Use the color camera index
(usually `0`) in `camera.index`.

## Camera calibration

Calibration is **only required for `pitch.mode: "pinhole"`**. It is not used by
`depth_anything`, which estimates relative depth directly with the model.

- `pinhole` loads `cameraCalibration/results/camera_matrix.npy` and
  `cameraCalibration/results/dist_coeffs.npy`.
- `depth_anything` needs no calibration files.
- Recalibrate if you change `camera.index`, `camera.width` or `camera.height`,
  because the intrinsics belong to that camera and resolution.

See [cameraCalibration/README.md](cameraCalibration/README.md) for how to use
the calibration app and what reprojection error to expect.

> **Nota:** una sola cámara no puede medir profundidad absoluta por sí sola. El
> backend `pinhole` usa la distancia focal calibrada y asume un ancho de mano
> canónico (`hand_real_width = 18 cm`) para fijar la escala:
> `D = f * W_real / W_imagen`. Por eso la profundidad es métrica solo si tu mano
> mide aproximadamente eso; si el tamaño aparente cambia por otra razón, la
> estimación también cambia. `depth_anything` no usa calibración ni escala métrica:
> devuelve profundidad relativa normalizada por frame.

## Tests

```bash
pytest tests/camera_test.py -v --tb=short --camera 0
python3 tests/handtracking_test.py
```

## HUD

The preview shows an SVG theremin drawn in perspective, with live pitch and
volume markers on its antennas plus a small panel with `FREQ`, `VOL` and `OUT`.
The HUD is scaled to the frame and anchored at the bottom center by default; change
`debug.hud_position` to move it. The `FREQ / VOL / OUT` panel is always drawn at
the top left so it does not cover the theremin or your hands.

Set `display.mode: "raw"` to show only the raw camera frame: no theremin HUD, no
hand landmarks and no values. Use `display.mode: "hud"` to draw everything again.

## Other libraries considered

- [pygame](https://www.pygame.org/news)
- [sounddevice](https://python-sounddevice.readthedocs.io/en/0.5.1/)
- [pyaudio](https://people.csail.mit.edu/hubert/pyaudio/)

## Links

- [MediaPipe Hands](https://mediapipe.readthedocs.io/en/latest/solutions/hands.html)
- [Making music with Python](https://medium.com/@stevehiehn/how-to-generate-music-with-python-the-basics-62e8ea9b99a5)
- [Theremin demo](https://www.youtube.com/watch?v=m_rmwcUREeY)
- [Theremin Python gist](https://gist.github.com/sahithyen/b20922c902620e5bd6fd926263a93836)
- [How a theremin works](https://splice.com/blog/how-theremin-works/)

## References

- Bartelt, T. L. M. (2001). *Industrial Control Electronics: Devices, Systems
  & Applications* (2nd ed.). Thomson Delmar Learning (p. 73).
- Casiez, G., Roussel, N., & Vogel, D. (2012). *1€ Filter: A Simple
  Speed-based Low-pass Filter for Noisy Input in Interactive Systems*. CHI 2012.

## TODO

- Web version (hence migrate all sound control, camera, etc. to browser managment).
- Status bars in [this style](https://www.linkedin.com/posts/siddhartha-reddy-054321341_python-opencv-mediapipe-activity-7360493210772598784-o8ko).
- If needed, use a Kalman filter for hand position and size (see p. 183, Simon Prince).
