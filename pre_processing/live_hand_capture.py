import cv2
import mediapipe as mp
import numpy as np
import os
import time
from collections import deque

# =======================
# Config
# =======================
CAM_INDEX = 0
WINDOW_LANDMARKS = "MediaPipe Landmarks"
WINDOW_PROC = "Silhouette & Edges"

# Output folders (created if missing)
SILHOUETTE_DIR = "hand_silhouette"
EDGE_DIR = "hand_edge"
FINGER_DIR = "hand_fingers"
os.makedirs(SILHOUETTE_DIR, exist_ok=True)
os.makedirs(EDGE_DIR, exist_ok=True)
os.makedirs(FINGER_DIR, exist_ok=True)

FINGER_COLORS = {
    "thumb": (0, 0, 255),     # Red
    "index": (0, 255, 0),     # Green
    "middle": (255, 0, 0),    # Blue
    "ring": (0, 255, 255),    # Yellow
    "pinky": (255, 0, 255),   # Purple
    "palm": (255, 255, 255)   # White
}

# Finger landmark groups
FINGER_LMS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20]
}

# MediaPipe config
MAX_HANDS = 1
MIN_DET_CONF = 0.60
MIN_TRK_CONF = 0.60

# Landmark drawing detail (tweak these for more/less detail)
DRAW_THICKNESS = 3         # smaller => more fine curves preserved
DRAW_CIRCLE_RADIUS = 2     # smaller => less blockiness at joints

# Morph kernels (tweak to adjust smoothness vs detail)
KERNEL = np.ones((3, 3), np.uint8)
DILATE_KERNEL = (5, 5)     # smaller => preserves more finger gaps
CLOSE_KERNEL = (7, 7)      # smaller => preserves more detail

# Saving controls
SAVE_ENABLED = True        # press 's' to toggle
SAVE_EVERY_N_FRAMES = 1    # save rate (1 = every detected frame)
MAX_SAVED_PER_SESSION = 0  # 0 = unlimited

# Optional simple FPS smoothing for overlay
FPS_AVG_OVER = 10

# =======================
# Helpers
# =======================
def timestamp_ms():
    return int(time.time() * 1000)

def build_mask_from_landmarks(img, hand_landmarks_list, mp_hands, mp_drawing):
    """
    Draws landmarks + connections onto a black 3-channel mask,
    then dilates & closes to create a filled hand region.
    """
    h, w, _ = img.shape
    mask_color = np.zeros((h, w, 3), dtype=np.uint8)

    for hand_landmarks in hand_landmarks_list:
        mp_drawing.draw_landmarks(
            mask_color,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 255),
                                   thickness=DRAW_THICKNESS,
                                   circle_radius=DRAW_CIRCLE_RADIUS)
        )

    # to single channel
    mask_gray = cv2.cvtColor(mask_color, cv2.COLOR_BGR2GRAY)
    # thicken strokes slightly, then close gaps to get a solid hand region
    mask_gray = cv2.dilate(mask_gray, np.ones(DILATE_KERNEL, np.uint8), iterations=2)
    mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_CLOSE, np.ones(CLOSE_KERNEL, np.uint8), iterations=1)
    return mask_gray

def build_finger_colored_mask(img, hand_landmarks, mp_hands):
    """
    Create a black canvas and color each finger + palm differently.
    """
    h, w, _ = img.shape
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Palm (just fill convex hull of landmarks excluding fingertips)
    palm_points = [
        (int(hand_landmarks.landmark[i].x * w), int(hand_landmarks.landmark[i].y * h))
        for i in [0, 1, 5, 9, 13, 17]  # wrist + base joints
    ]
    if len(palm_points) >= 3:
        cv2.fillConvexPoly(mask, np.array(palm_points), FINGER_COLORS["palm"])

    # Fingers
    for finger, ids in FINGER_LMS.items():
        points = [(int(hand_landmarks.landmark[i].x * w), int(hand_landmarks.landmark[i].y * h)) for i in ids]
        for i in range(len(points) - 1):
            cv2.line(mask, points[i], points[i+1], FINGER_COLORS[finger], thickness=12)
        # fingertip circle
        cv2.circle(mask, points[-1], 12, FINGER_COLORS[finger], -1)

    return mask

def preprocess_to_silhouette_and_edges(frame_bgr, hand_mask):
    """
    Applies your stable preprocessing:
    - mask the hand region
    - grayscale + median blur
    - Otsu threshold
    - morphological close/open cleanup
    - edges derived from silhouette (dilated)
    """
    hand_only = cv2.bitwise_and(frame_bgr, frame_bgr, mask=hand_mask)

    gray = cv2.cvtColor(hand_only, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)

    # Otsu threshold for silhouette
    _, silhouette = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Cleanup
    silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_OPEN, KERNEL, iterations=1)

    # Edges from silhouette (stable)
    edges = cv2.Canny(silhouette, 3, 100)
    edges = cv2.dilate(edges, KERNEL, iterations=1)

    return silhouette, edges

# =======================
# Main
# =======================
def main():
    SAVE_ENABLED = True
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("❌ Could not open camera.")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,        # tracking mode for speed
        max_num_hands=MAX_HANDS,
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRK_CONF
    )

    cv2.namedWindow(WINDOW_LANDMARKS, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_PROC, cv2.WINDOW_NORMAL)

    frame_count = 0
    saved_count = 0
    fps_times = deque(maxlen=FPS_AVG_OVER)
    last_tick = time.time()

    print("▶ Live capture started.")
    print("   Keys: [q] quit  |  [s] toggle saving  |  [h] hide/show help")

    show_help = True

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        vis = frame.copy()

        # MediaPipe detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        silhouette, edges, finger_mask = None, None, None
        detected = results.multi_hand_landmarks is not None

        if detected:
            # Draw landmarks on the visualization frame
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    vis, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=2, circle_radius=2)
                )

            # Build mask from landmarks and preprocess
            hand_mask = build_mask_from_landmarks(frame, results.multi_hand_landmarks, mp_hands, mp_drawing)
            silhouette, edges = preprocess_to_silhouette_and_edges(frame, hand_mask)
            # Build finger-colored mask (only first hand for now)
            finger_mask = build_finger_colored_mask(frame, results.multi_hand_landmarks[0], mp_hands)


        # Compose the processed window image
        if silhouette is not None and edges is not None and finger_mask is not None:
            silu_bgr = cv2.cvtColor(silhouette, cv2.COLOR_GRAY2BGR)
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            proc_vis = np.hstack([silu_bgr, edges_bgr, finger_mask])
        else:
            # If no hand detected, show a black canvas with a message
            h, w, _ = frame.shape
            left = np.zeros((h, w, 3), dtype=np.uint8)
            right = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(left, "No hand detected", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
            proc_vis = np.hstack([left, right])

        # FPS
        now = time.time()
        dt = now - last_tick
        last_tick = now
        fps_times.append(1.0 / max(dt, 1e-6))
        fps = sum(fps_times) / len(fps_times)

        # Overlays
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        status = f"SAVE: {'ON' if SAVE_ENABLED else 'OFF'} | Every {SAVE_EVERY_N_FRAMES} frame(s)"
        cv2.putText(vis, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 255, 180), 2)
        if show_help:
            cv2.putText(vis, "[q] quit  [s] toggle save  [h] toggle help",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 255), 2)

        # Show windows
        cv2.imshow(WINDOW_LANDMARKS, vis)
        cv2.imshow(WINDOW_PROC, proc_vis)

        # Save if enabled and a hand is detected
        if detected and SAVE_ENABLED and (frame_count % SAVE_EVERY_N_FRAMES == 0) and (MAX_SAVED_PER_SESSION == 0 or saved_count < MAX_SAVED_PER_SESSION):
            t = timestamp_ms()
            silu_path = os.path.join(SILHOUETTE_DIR, f"silhouette_{t}.png")
            edge_path = os.path.join(EDGE_DIR, f"edge_{t}.png")

            # Only save when we actually have images
            if silhouette is not None and edges is not None and finger_mask is not None:
                cv2.imwrite(silu_path, silhouette)
                cv2.imwrite(edge_path, edges)
                finger_path = os.path.join(FINGER_DIR, f"fingers_{t}.png")
                cv2.imwrite(finger_path, finger_mask)
                saved_count += 1

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            SAVE_ENABLED = not SAVE_ENABLED
        elif key == ord('h'):
            show_help = not show_help

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print(f"⏹ Done. Saved {saved_count} pairs to:")
    print(f"   - {SILHOUETTE_DIR}")
    print(f"   - {EDGE_DIR}")

if __name__ == "__main__":
    main()

