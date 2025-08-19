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

# Dataset directory
DATASET_DIR = "datasets"
os.makedirs(DATASET_DIR, exist_ok=True)

# Classes (skip J and Z since dynamic)
CLASSES = list("ABCDEFGHIKLMNOPQRSTUVWXY")

# MediaPipe config
MAX_HANDS = 2
MIN_DET_CONF = 0.60
MIN_TRK_CONF = 0.60

# Landmark drawing detail
DRAW_THICKNESS = 3
DRAW_CIRCLE_RADIUS = 2

# Morph kernels
KERNEL = np.ones((3, 3), np.uint8)
DILATE_KERNEL = (5, 5)
CLOSE_KERNEL = (7, 7)

# Capture timing
CAPTURE_DURATION = 15   # seconds per letter
BREAK_DURATION = 20     # seconds break between letters

# =======================
# Helpers
# =======================
def timestamp_ms():
    return int(time.time() * 1000)

def build_mask_from_landmarks(img, hand_landmarks_list, mp_hands, mp_drawing):
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

    mask_gray = cv2.cvtColor(mask_color, cv2.COLOR_BGR2GRAY)
    mask_gray = cv2.dilate(mask_gray, np.ones(DILATE_KERNEL, np.uint8), iterations=2)
    mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_CLOSE, np.ones(CLOSE_KERNEL, np.uint8), iterations=1)
    return mask_gray

def preprocess_to_silhouette_and_edges(frame_bgr, hand_mask):
    hand_only = cv2.bitwise_and(frame_bgr, frame_bgr, mask=hand_mask)

    gray = cv2.cvtColor(hand_only, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)

    _, silhouette = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_OPEN, KERNEL, iterations=1)

    edges = cv2.Canny(silhouette, 3, 100)
    edges = cv2.dilate(edges, KERNEL, iterations=1)

    return silhouette, edges

# =======================
# Main
# =======================
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("❌ Could not open camera.")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRK_CONF
    )

    cv2.namedWindow(WINDOW_LANDMARKS, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_PROC, cv2.WINDOW_NORMAL)

    fps_times = deque(maxlen=10)
    last_tick = time.time()

    for cls in CLASSES:
        print(f"\n▶ Prepare to show gesture for: {cls}")
        time.sleep(3)

        # Class directory
        class_dir = os.path.join(DATASET_DIR, cls)
        silhouette_dir = os.path.join(class_dir, "silhouette")
        edge_dir = os.path.join(class_dir, "edge")
        os.makedirs(silhouette_dir, exist_ok=True)
        os.makedirs(edge_dir, exist_ok=True)

        start_time = time.time()
        saved_count = 0

        while (time.time() - start_time) < CAPTURE_DURATION:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            vis = frame.copy()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            silhouette, edges = None, None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        vis, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=2, circle_radius=2)
                    )

                hand_mask = build_mask_from_landmarks(frame, results.multi_hand_landmarks, mp_hands, mp_drawing)
                silhouette, edges = preprocess_to_silhouette_and_edges(frame, hand_mask)

            if silhouette is not None and edges is not None:
                silu_bgr = cv2.cvtColor(silhouette, cv2.COLOR_GRAY2BGR)
                edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                proc_vis = np.hstack([silu_bgr, edges_bgr])

                t = timestamp_ms()
                cv2.imwrite(os.path.join(silhouette_dir, f"{t}.png"), silhouette)
                cv2.imwrite(os.path.join(edge_dir, f"{t}.png"), edges)
                saved_count += 1
            else:
                h, w, _ = frame.shape
                proc_vis = np.zeros((h, w * 2, 3), dtype=np.uint8)
                cv2.putText(proc_vis, "No hand detected", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)

            # FPS overlay
            now = time.time()
            fps_times.append(1.0 / max(now - last_tick, 1e-6))
            last_tick = now
            fps = sum(fps_times) / len(fps_times)
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            cv2.imshow(WINDOW_LANDMARKS, vis)
            cv2.imshow(WINDOW_PROC, proc_vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                hands.close()
                return

        print(f"✅ Saved {saved_count} pairs for '{cls}'")

        if cls != CLASSES[-1]:
            print(f"⏸ Break for {BREAK_DURATION} seconds before next letter...")
            time.sleep(BREAK_DURATION)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("🎉 Dataset collection complete!")

if __name__ == "__main__":
    main()

