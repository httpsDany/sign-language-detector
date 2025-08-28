# filters.py
import cv2
import numpy as np

def build_mask_from_landmarks(img, hand_landmarks_list, mp_hands, mp_drawing):
    h, w, _ = img.shape
    mask_color = np.zeros((h, w, 3), dtype=np.uint8)
    for hand_landmarks in hand_landmarks_list:
        mp_drawing.draw_landmarks(
            mask_color, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=2)
        )
    mask_gray = cv2.cvtColor(mask_color, cv2.COLOR_BGR2GRAY)
    mask_gray = cv2.dilate(mask_gray, np.ones((5, 5), np.uint8), iterations=2)
    mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    return mask_gray

def preprocess_to_silhouette_and_edges(frame_bgr, hand_mask):
    hand_only = cv2.bitwise_and(frame_bgr, frame_bgr, mask=hand_mask)
    gray = cv2.cvtColor(hand_only, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    _, silhouette = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.Canny(silhouette, 3, 100)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return silhouette, edges

def build_finger_colored_mask(img, hand_landmarks):
    h, w, _ = img.shape
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    finger_colors = {
        "thumb": (255, 0, 0),
        "index": (0, 255, 0),
        "middle": (0, 0, 255),
        "ring": (255, 255, 0),
        "pinky": (255, 0, 255),
        "palm": (255, 255, 255)
    }
    finger_lms = {
        "thumb": [1, 2, 3, 4],
        "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20]
    }
    palm_points = [(int(hand_landmarks.landmark[i].x * w), int(hand_landmarks.landmark[i].y * h)) for i in [0,1,5,9,13,17]]
    if len(palm_points) >= 3:
        cv2.fillConvexPoly(mask, np.array(palm_points), finger_colors["palm"])
    for finger, ids in finger_lms.items():
        pts = [(int(hand_landmarks.landmark[i].x * w), int(hand_landmarks.landmark[i].y * h)) for i in ids]
        for i in range(len(pts)-1):
            cv2.line(mask, pts[i], pts[i+1], finger_colors[finger], thickness=12)
        cv2.circle(mask, pts[-1], 12, finger_colors[finger], -1)
    return mask

