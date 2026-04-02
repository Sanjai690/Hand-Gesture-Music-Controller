import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import librosa
import time
import os

# ----------------- USER SETTINGS -----------------
AUDIO_FILES = ["music1.wav", "music2.wav"]  # Add your music files here
SPEED_RANGE = (0.5, 2.0)  # right hand vertical
PITCH_RANGE = (0.8, 1.25)  # left hand vertical
VOLUME_RANGE = (0.0, 1.0)  # min to max volume
CAM_WIDTH, CAM_HEIGHT = 1280, 720  # ✅ Bigger camera size
# -------------------------------------------------

# Check if audio files exist
available_files = []
for file in AUDIO_FILES:
    if os.path.exists(file):
        available_files.append(file)
    else:
        print(f"Warning: Audio file '{file}' not found.")

if not available_files:
    raise FileNotFoundError("No valid audio files found.")

# Current song tracking
current_song_index = 0
current_audio_file = available_files[current_song_index]

# Load initial audio
y, sr = librosa.load(current_audio_file, sr=None, mono=True)

# Shared state variables
speed_mult = 1.0
pitch_mult = 1.0
volume_val = 0.5
position = 0
paused = False  # <-- new state


# Audio callback for continuous playback
def callback(outdata, frames, time_info, status):
    global position, speed_mult, pitch_mult, volume_val, paused
    if status:
        print(status)

    if paused:  # silence output if paused
        outdata.fill(0)
        return

    rate = sr * speed_mult * pitch_mult
    grab = int(frames * (rate / sr))

    if position + grab >= len(y):
        position = 0  # loop back to start

    chunk = y[position:position + grab]

    if len(chunk) > 0:
        resampled = librosa.resample(chunk, orig_sr=rate, target_sr=sr)
    else:
        resampled = np.zeros(frames)

    if len(resampled) < frames:
        out = np.zeros(frames)
        out[:len(resampled)] = resampled
    else:
        out = resampled[:frames]

    outdata[:, 0] = out * volume_val
    position += grab


# Start audio stream
stream = sd.OutputStream(samplerate=sr, channels=1, callback=callback, blocksize=1024)
stream.start()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# ✅ Set camera resolution (high-res if supported)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

# ✅ Make OpenCV window resizable
cv2.namedWindow("Hand Music Controller", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hand Music Controller", CAM_WIDTH, CAM_HEIGHT)


def map_range(value, in_min, in_max, out_min, out_max):
    return out_min + (float(value - in_min) / float(in_max - in_min)) * (out_max - out_min)


def is_hand_closed(hand_landmarks):
    """Check if hand is closed (fist) using finger tip positions."""
    tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
    pips = [3, 6, 10, 14, 18]  # Lower joints
    closed_count = 0
    for tip, pip in zip(tips[1:], pips[1:]):  # ignore thumb for simplicity
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y:
            closed_count += 1
    return closed_count >= 3  # if 3 or more fingers bent, call it closed


def detect_palm_flip(hand_landmarks):
    """Detect palm flip motion by tracking thumb position relative to other fingers."""
    # Use thumb and pinky to detect hand rotation/flip
    thumb_tip = hand_landmarks.landmark[4]  # Thumb tip
    pinky_tip = hand_landmarks.landmark[20]  # Pinky tip
    wrist = hand_landmarks.landmark[0]  # Wrist

    # Calculate relative positions
    thumb_x = thumb_tip.x
    pinky_x = pinky_tip.x

    # Determine if thumb is on left or right side of pinky
    if thumb_x < pinky_x:
        return "left"  # Thumb on left side
    else:
        return "right"  # Thumb on right side


def switch_to_next_song():
    """Switch to the next song in the playlist."""
    global current_song_index, current_audio_file, y, sr, position, stream

    # Stop current stream
    stream.stop()
    stream.close()

    current_song_index = (current_song_index + 1) % len(available_files)
    current_audio_file = available_files[current_song_index]

    print(f"🎵 Switching to: {current_audio_file}")

    # Load new audio
    y, sr = librosa.load(current_audio_file, sr=None, mono=True)
    position = 0  # Reset position for new song

    # Restart stream with new audio
    stream = sd.OutputStream(samplerate=sr, channels=1, callback=callback, blocksize=1024)
    stream.start()


def draw_control_zones(frame, speed_val, pitch_val, volume_val, is_paused):
    """Draw clean white control lines exactly like in the AR image."""
    h, w = frame.shape[:2]

    # Volume control - moved to top center
    center_x = w // 2
    vol_bar_width = int(w * 0.25)  # 25% of screen width
    vol_bar_height = 15
    vol_y = 50

    # Volume background bar (horizontal at top center)
    cv2.rectangle(frame, (center_x - vol_bar_width // 2, vol_y),
                  (center_x + vol_bar_width // 2, vol_y + vol_bar_height), (100, 100, 100), 2)

    # Volume level fill
    fill_width = int(volume_val * vol_bar_width)
    if fill_width > 0:
        cv2.rectangle(frame, (center_x - vol_bar_width // 2 + 2, vol_y + 2),
                      (center_x - vol_bar_width // 2 + fill_width, vol_y + vol_bar_height - 2), (255, 255, 255), -1)

    # Volume label and value
    cv2.putText(frame, "Volume", (center_x - 30, vol_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"{int(volume_val * 100)}", (center_x + vol_bar_width // 2 + 10, vol_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Speed control - right side (made shorter width)
    speed_x = w - 60
    bar_height = int(h * 0.4)  # shorter height
    bar_top = int(h * 0.3)
    bar_bottom = bar_top + bar_height
    speed_normalized = (speed_val - SPEED_RANGE[0]) / (SPEED_RANGE[1] - SPEED_RANGE[0])

    # Speed background bar (thinner)
    cv2.rectangle(frame, (speed_x - 8, bar_top), (speed_x + 8, bar_bottom), (100, 100, 100), 2)

    # Speed level fill
    speed_fill = int(speed_normalized * bar_height)
    if speed_fill > 0:
        cv2.rectangle(frame, (speed_x - 6, bar_bottom - speed_fill),
                      (speed_x + 6, bar_bottom), (255, 255, 255), -1)

    # Speed label and value
    cv2.putText(frame, "Speed", (speed_x - 25, bar_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"{speed_val:.1f}", (speed_x - 15, bar_bottom + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Frequency control - left side (made shorter width)
    freq_x = 60
    freq_normalized = (pitch_val - PITCH_RANGE[0]) / (PITCH_RANGE[1] - PITCH_RANGE[0])

    # Frequency background bar (thinner)
    cv2.rectangle(frame, (freq_x - 8, bar_top), (freq_x + 8, bar_bottom), (100, 100, 100), 2)

    # Frequency level fill
    freq_fill = int(freq_normalized * bar_height)
    if freq_fill > 0:
        cv2.rectangle(frame, (freq_x - 6, bar_bottom - freq_fill),
                      (freq_x + 6, bar_bottom), (255, 255, 255), -1)

    # Frequency label and value
    cv2.putText(frame, "Frequency", (freq_x - 30, bar_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"{pitch_val:.1f}", (freq_x - 15, bar_bottom + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Simple status text with current song info
    status_text = "PLAYING" if not is_paused else "PAUSED"
    cv2.putText(frame, status_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display current song name
    song_name = os.path.basename(current_audio_file).replace('.wav', '').replace('.mp3', '')
    cv2.putText(frame, f"Song: {song_name}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    # Display song counter
    cv2.putText(frame, f"{current_song_index + 1}/{len(available_files)}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)


print(
    f"🎵 Running: Right=Speed, Left=Pitch, Distance=Volume. Fists=Pause, Palms=Play. Flip hand to switch songs. Press 'q' to quit.")
print(f"🎶 Current song: {current_audio_file}")

# Hand flip tracking
previous_hand_orientations = {}  # Track previous hand orientations
flip_cooldown = {}  # Prevent rapid song switching
last_flip_time = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # ✅ Flip the camera horizontally so right=right and left=left
        frame = cv2.flip(frame, 1)

        # ✅ Resize frame to bigger display size
        frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))

        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        speed_val = speed_mult
        pitch_val = pitch_mult
        vol_val = volume_val

        hand_states = []  # open/closed states

        if results.multi_hand_landmarks and results.multi_handedness:
            hand_positions = {}
            for handedness, lm in zip(results.multi_handedness, results.multi_hand_landmarks):
                label = handedness.classification[0].label
                mean_y = np.mean([p.y for p in lm.landmark])
                mean_x = np.mean([p.x for p in lm.landmark])
                hand_positions[label] = (mean_x, mean_y)

                # Detect palm flip for song switching
                current_orientation = detect_palm_flip(lm)
                import time

                current_time = time.time()

                # Check if hand orientation changed (flip detected)
                if label in previous_hand_orientations:
                    prev_orientation = previous_hand_orientations[label]

                    # Detect flip: thumb switches from left to right or right to left
                    if ((prev_orientation == "left" and current_orientation == "right") or
                            (prev_orientation == "right" and current_orientation == "left")):

                        # Check cooldown to prevent rapid switching
                        if current_time - last_flip_time > 1.5:  # 1.5 second cooldown
                            switch_to_next_song()
                            last_flip_time = current_time
                            print(f"🔄 Palm flip detected! Switched to: {current_audio_file}")

                # Update previous orientation
                previous_hand_orientations[label] = current_orientation

                # Detect open/closed
                closed = is_hand_closed(lm)
                hand_states.append(closed)

                if label == "Right":
                    speed_val = map_range(mean_y, 1.0, 0.0, SPEED_RANGE[0], SPEED_RANGE[1])
                elif label == "Left":
                    pitch_val = map_range(mean_y, 1.0, 0.0, PITCH_RANGE[0], PITCH_RANGE[1])

            if "Right" in hand_positions and "Left" in hand_positions:
                dist = abs(hand_positions["Right"][0] - hand_positions["Left"][0])
                vol_val = np.clip(map_range(dist, 0.05, 0.5, VOLUME_RANGE[0], VOLUME_RANGE[1]), 0.0, 1.0)

                # Draw equalizer-style vertical bars like in the image
                right_pos = (int(hand_positions["Right"][0] * w), int(hand_positions["Right"][1] * h))
                left_pos = (int(hand_positions["Left"][0] * w), int(hand_positions["Left"][1] * h))

                # Only draw equalizer bars if music is playing
                if not paused:
                    import time

                    current_time = time.time()

                    # Calculate the line between hands
                    hand_distance = ((right_pos[0] - left_pos[0]) * 2 + (right_pos[1] - left_pos[1])) * 0.5

                    # Create equalizer bars along the line between hands
                    num_bars = 20  # more bars for closer spacing
                    bar_spacing = max(8, int(hand_distance / num_bars))

                    for i in range(num_bars):
                        # Calculate position along the line between hands
                        t = i / (num_bars - 1) if num_bars > 1 else 0
                        bar_x = int(left_pos[0] + t * (right_pos[0] - left_pos[0]))
                        bar_y = int(left_pos[1] + t * (right_pos[1] - left_pos[1]))

                        # Create dynamic bar heights based on music parameters
                        # Each bar has different frequency response like real equalizer
                        frequency_factor = (i / num_bars) * 2 * np.pi  # spread across frequency spectrum

                        # Volume creates base amplitude
                        base_height = volume_val * 50

                        # Speed affects animation rate
                        speed_animation = np.sin(current_time * speed_mult * 4 + frequency_factor) * 18

                        # Pitch affects frequency response - higher pitch = more high frequency bars
                        pitch_response = np.sin(frequency_factor * pitch_mult + current_time * 2) * 12

                        # Individual bar oscillation for realistic equalizer effect
                        individual_pulse = np.sin(current_time * 6 + i * 0.4) * 8

                        # Combine all effects for bar height
                        total_height = int(base_height + speed_animation + pitch_response + individual_pulse)
                        total_height = max(8, min(total_height, 80))  # limit bar height

                        # Create the vertical equalizer bar
                        bar_top_y = bar_y - total_height // 2
                        bar_bottom_y = bar_y + total_height // 2

                        # Bar thickness - thick but not too thick (fixed optimal size)
                        bar_thickness = max(4, min(int(volume_val * 6 + 2), 8))  # thickness range 4-8

                        # Draw the vertical equalizer bar (like in the image)
                        cv2.line(frame, (bar_x, bar_top_y), (bar_x, bar_bottom_y),
                                 (255, 255, 255), bar_thickness)

                        # Add small caps at top and bottom for professional look
                        cap_width = bar_thickness + 2
                        cv2.line(frame, (bar_x - cap_width // 2, bar_top_y),
                                 (bar_x + cap_width // 2, bar_top_y), (255, 255, 255), 2)
                        cv2.line(frame, (bar_x - cap_width // 2, bar_bottom_y),
                                 (bar_x + cap_width // 2, bar_bottom_y), (255, 255, 255), 2)

                    # Add connecting base line (subtle)
                    cv2.line(frame, left_pos, right_pos, (100, 100, 100), 1)

                    # Add frequency labels like in the image (optional)
                    mid_x = (left_pos[0] + right_pos[0]) // 2
                    cv2.putText(frame, "EQUALIZER", (mid_x - 40, left_pos[1] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                else:
                    # Music is paused - show static equalizer bars
                    num_bars = 20  # same number as when playing
                    hand_distance = ((right_pos[0] - left_pos[0]) * 2 + (right_pos[1] - left_pos[1])) * 0.5

                    for i in range(num_bars):
                        t = i / (num_bars - 1) if num_bars > 1 else 0
                        bar_x = int(left_pos[0] + t * (right_pos[0] - left_pos[0]))
                        bar_y = int(left_pos[1] + t * (right_pos[1] - left_pos[1]))

                        # Static small bars when paused
                        static_height = 12
                        cv2.line(frame, (bar_x, bar_y - static_height // 2),
                                 (bar_x, bar_y + static_height // 2), (100, 100, 100), 3)

                    # Pause indicator
                    mid_x = (left_pos[0] + right_pos[0]) // 2
                    mid_y = (left_pos[1] + right_pos[1]) // 2
                    cv2.rectangle(frame, (mid_x - 8, mid_y - 6), (mid_x - 2, mid_y + 6), (150, 150, 150), -1)
                    cv2.rectangle(frame, (mid_x + 2, mid_y - 6), (mid_x + 8, mid_y + 6), (150, 150, 150), -1)

                # Reference line
                cv2.line(frame, left_pos, right_pos, (60, 60, 60), 1)

        # Pause/Play detection
        if len(hand_states) == 2:
            if all(hand_states):  # both closed
                paused = True
            elif not any(hand_states):  # both open
                paused = False

        # Update shared values
        speed_mult = speed_val
        pitch_mult = pitch_val
        volume_val = vol_val

        # Draw clean control zones like in the image
        draw_control_zones(frame, speed_mult, pitch_mult, volume_val, paused)

        # Draw simple hand landmarks and indicators
        if results.multi_hand_landmarks and results.multi_handedness:
            for handedness, lm in zip(results.multi_handedness, results.multi_hand_landmarks):
                label = handedness.classification[0].label

                # Draw clean hand landmarks
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                                       mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2))

                # Simple hand position indicators
                mean_x = int(np.mean([p.x for p in lm.landmark]) * w)
                mean_y = int(np.mean([p.y for p in lm.landmark]) * h)

                # Clean white circles for hand indicators
                cv2.circle(frame, (mean_x, mean_y), 8, (255, 255, 255), 2)

                if label == "Right":
                    cv2.putText(frame, "Speed", (mean_x - 25, mean_y - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                elif label == "Left":
                    cv2.putText(frame, "Pitch", (mean_x - 25, mean_y - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Update shared values
        speed_mult = speed_val
        pitch_mult = pitch_val
        volume_val = vol_val

        cv2.imshow("Hand Music Controller", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    stream.stop()
    stream.close()
    cap.release()
    cv2.destroyAllWindows()