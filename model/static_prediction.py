import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms as T
from PIL import Image
from collections import deque, Counter

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pre_processing.filters import (
    build_mask_from_landmarks,
    preprocess_to_silhouette_and_edges,
    build_finger_colored_mask,
)

# =======================
# CONFIG
# =======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
CAM_INDEX = 0
CLASSES = list("ABCDEFGHIKLMNOPQRSTUVWXY")  # A-Y, no J/Z
TEMPORAL_WINDOW = 10  # number of frames for temporal smoothing
EDGE_WEIGHT = 0.6
FINGER_WEIGHT = 0.4

# =======================
# TRANSFORMS
# =======================
transform_gray = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])
transform_rgb = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# =======================
# MODEL BUILDER
# =======================
def build_model(num_classes, in_channels=3):
    model = models.efficientnet_b0(weights=None)
    if in_channels == 1:
        original_conv = model.features[0][0]
        new_conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        model.features[0][0] = new_conv
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

# =======================
# PREDICTION
# =======================
def predict(model, in_channels, img):
    if in_channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pil_img = Image.fromarray(img)
        tensor = transform_gray(pil_img).unsqueeze(0).to(DEVICE)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        tensor = transform_rgb(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        _, pred = torch.max(outputs, 1)
    return CLASSES[pred.item()]

# =======================
# TEMPORAL SMOOTHING
# =======================
class TemporalSmoother:
    def __init__(self, max_len=5):
        self.buffer = deque(maxlen=max_len)
    def update(self, val):
        self.buffer.append(val)
        most_common = Counter(self.buffer).most_common(1)
        return most_common[0][0] if most_common else None

# =======================
# MAIN LOOP
# =======================
def main():
    # Load models
    edge_model = build_model(len(CLASSES), in_channels=1).to(DEVICE)
    edge_model.load_state_dict(torch.load("hand_model_edge.pth", map_location=DEVICE))
    edge_model.eval()

    fingers_model = build_model(len(CLASSES), in_channels=3).to(DEVICE)
    fingers_model.load_state_dict(torch.load("hand_model_fingers.pth", map_location=DEVICE))
    fingers_model.eval()

    cap = cv2.VideoCapture(CAM_INDEX)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)

    temporal_smoother = TemporalSmoother(max_len=TEMPORAL_WINDOW)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        pred_edge, pred_fingers, combined_pred, smoothed_pred = None, None, None, None

        if results.multi_hand_landmarks:
            hand_mask = build_mask_from_landmarks(frame, results.multi_hand_landmarks, mp_hands, mp_drawing)
            silhouette, edges = preprocess_to_silhouette_and_edges(frame, hand_mask)
            finger_mask = build_finger_colored_mask(frame, results.multi_hand_landmarks[0])

            pred_edge = predict(edge_model, 1, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
            pred_fingers = predict(fingers_model, 3, finger_mask)

            # Weighted voting
            votes = {}
            for c in CLASSES:
                votes[c] = 0
            votes[pred_edge] += EDGE_WEIGHT
            votes[pred_fingers] += FINGER_WEIGHT
            combined_pred = max(votes, key=votes.get)

            # Temporal smoothing
            smoothed_pred = temporal_smoother.update(combined_pred)

        # Display all predictions
        if pred_edge: cv2.putText(frame, f"EDGE: {pred_edge}", (10,40), cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),2)
        if pred_fingers: cv2.putText(frame, f"FINGERS: {pred_fingers}", (10,80), cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,0,0),2)
        if combined_pred: cv2.putText(frame, f"COMBINED: {combined_pred}", (10,120), cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,255),2)
        if smoothed_pred: cv2.putText(frame, f"SMOOTHED: {smoothed_pred}", (10,160), cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),2)

        cv2.imshow("Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()

