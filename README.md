# cv-theremin
A handtracking and 2D camera based digital theremin

*A work in progress*

<h2>Tools used</h2>

- Python v3.10.12 for programming
- `opencv_contrib_python` for camera usage and frames manipulation
- `mediapipe` for hand tracking
- `pipreqs` for listing all requirements

<h2>Usage of the theremin in a virtual environment</h2>

Install venv, e.g.
`sudo apt install python3.10-venv `

Create a virtual environment:
`python3 -m venv test`

Activate it:
`source ./bin/activate`

Open VS Code:
`code .`

Install required dependencies:
`python3 -m pip install -r requirements.txt`

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
