#Hand Gesture Music Controller 

This workspace contains two webcam-based interactive Python demos built with OpenCV and MediaPipe Hands.

- `gampro.py`: a gesture boxing game where punch-like hand motion damages the enemy.
- `teck.py`: a hand-controlled music visualizer/player that maps hand position and gestures to playback controls.

## Requirements

- Python 3.10 or newer
- A working webcam
- Working speakers or headphones for the music controller
- The following Python packages:
  - `opencv-python`
  - `mediapipe`
  - `pygame`
  - `numpy`
  - `sounddevice`
  - `librosa`

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install opencv-python mediapipe pygame numpy sounddevice librosa
```

If `sounddevice` fails to install on Windows, make sure PortAudio support is available for your Python environment.

## Run

Run the demo you want:

```bash
python gampro.py
```

```bash
python teck.py
```

## Demo Details

### `gampro.py`

- Opens a 900 x 500 Pygame window.
- Uses the webcam and MediaPipe Hands to detect punch-like wrist motion.
- A successful punch reduces the enemy health bar.
- The game ends when either fighter reaches zero health.

### `teck.py`

- Uses the webcam to track both hands in real time.
- Maps the right hand to speed, the left hand to pitch/frequency, and the distance between hands to volume.
- Uses fist and palm states to pause or resume playback.
- A palm flip switches to the next available audio file.

## Controls

### `gampro.py`

- Move your hand quickly forward to throw a punch.
- Close the window to exit.

### `teck.py`

- Move the right hand vertically to change speed.
- Move the left hand vertically to change pitch.
- Change the distance between both hands to change volume.
- Make both hands into fists to pause.
- Open both hands to play.
- Flip a hand orientation to switch songs.
- Press `q` to quit.

## Audio Files

`teck.py` looks for these audio files in the project root:

- `music1.wav`
- `music2.wav`

At least one of them must exist before starting the music controller.

## Notes

- The demos are designed to run from the project root so relative asset paths resolve correctly.
- If the webcam does not open, check that no other application is using it.
