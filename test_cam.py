import cv2
import mediapipe as mp
import time

def classify_gesture(hand_lm):
    """Simple gesture classifier based on finger states."""
    lm = hand_lm.landmark

    # Finger tips and PIP joints (for open/closed detection)
    tips = [4, 8, 12, 16, 20]   # Thumb, Index, Middle, Ring, Pinky tips
    pips = [3, 6, 10, 14, 18]   # PIP joints

    fingers_up = []
    for tip, pip in zip(tips[1:], pips[1:]):
        fingers_up.append(lm[tip].y < lm[pip].y)

    # Thumb
    fingers_up.insert(0, lm[4].x < lm[3].x)

    count = sum(fingers_up)

    if count == 0: return "Fist ✊", 0.9
    if count == 5: return "Open Hand 🖐", 0.9
    if fingers_up[1] and not any(fingers_up[2:]): return "Point ☝️", 0.85
    if fingers_up[1] and fingers_up[2] and not any(fingers_up[3:]): return "Peace ✌️", 0.85
    if fingers_up[0] and not any(fingers_up[1:]): return "Thumbs Up 👍", 0.85
    return f"{count} fingers", 0.7

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)
# Just reading a single frame if possible
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No webcam 0")
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("No webcam 1")
else:
    for i in range(10):
        ret, frame = cap.read()
        if ret:
            print("Webcam read successful. Shape:", frame.shape)
            break
        time.sleep(0.1)
