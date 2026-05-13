import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

print("Starting test...")
model_path = os.path.abspath('hand_landmarker.task')
print(f"Model path: {model_path}")

try:
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )
    detector = vision.HandLandmarker.create_from_options(options)
    print("Detector created!")

    # Create dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    results = detector.detect(mp_image)
    print("Detection successful!", results)
except Exception as e:
    print("Error:", e)
