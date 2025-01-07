# cv-theremin
A handtracking and 2D camera based digital theremin

*A work in progress*

<h2>Tools used</h2>

- [Python v3.10.12](https://docs.python-guide.org/starting/install3/linux/) for programming
- [`opencv_contrib_python`](https://github.com/opencv/opencv-python?tab=readme-ov-file#installation-and-usage) for camera usage and frames manipulation
- [`mediapipe`](https://github.com/google-ai-edge/mediapipe) for hand tracking
- [`pyo`](https://github.com/belangeo/pyo) for audio managment
- [`pipreqs`](https://github.com/bndr/pipreqs) for listing all requirements
- [`skfuzzy`](https://pythonhosted.org/scikit-fuzzy/) for fuzzy sets implementation of tone control

<h2>Usage of the theremin in a virtual environment</h2>

Install venv, e.g.
`sudo apt install python3.10-venv `

Create a virtual environment:
`python3 -m venv test`

Activate it:
`source ./bin/activate`

Open VS Code:
`code .`

Be sure to use the correct Python interpreter, e.g.:
`../bin/python3`

Install required dependencies:
`python3 -m pip install -r requirements.txt`

Run main program:
`python3 main.py`

<h2>To run tests</h2>

`pytest tests/camera_test.py -v --tb=short --camera 0`

`python3 tests/handtracking_test.py`

<h2>Other libraries considered but not used</h2>

- [pygame](https://www.pygame.org/news) 
- [sounddevice](https://python-sounddevice.readthedocs.io/en/0.5.1/)
- [pyaudio](https://people.csail.mit.edu/hubert/pyaudio/) 

<h2>Interesting links</h2>

- https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
- https://medium.com/@stevehiehn/how-to-generate-music-with-python-the-basics-62e8ea9b99a5
- https://www.youtube.com/watch?v=m_rmwcUREeY
- https://gist.github.com/sahithyen/b20922c902620e5bd6fd926263a93836
- https://splice.com/blog/how-theremin-works/

<h2>TODO</h2>
- Reference fuzzy sets book
