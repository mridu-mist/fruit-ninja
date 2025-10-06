import cv2
import mediapipe as mp
import random
import math
import time
import pygame
import os
import numpy as np

# ------------------------------
# 🎵 Safe Sound Setup
# ------------------------------
sound_enabled = False
slice_sound = None
try:
    pygame.mixer.init()
    if os.path.exists("slice.wav"):
        try:
            slice_sound = pygame.mixer.Sound("slice.wav")
            sound_enabled = True
        except Exception as e:
            print("⚠️ Could not load slice.wav:", e)
    else:
        print("⚠️ slice.wav not found – running without sound.")
except Exception as e:
    print("⚠️ pygame.mixer.init() failed — audio disabled.", e)
    sound_enabled = False

def play_slice_sound():
    if sound_enabled and slice_sound:
        try:
            slice_sound.play()
        except Exception:
            pass

# ------------------------------
# 🎨 Load Images (with checks)
# ------------------------------
def load_image_rgba(path, scale=0.1):
    if not os.path.exists(path):
        print(f"⚠️ {path} not found.")
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or BGRA
    if img is None:
        print(f"⚠️ Failed to load {path}.")
        return None

    # If image does not have alpha channel, add fully opaque alpha
    if img.shape[2] == 3:
        alpha = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
        img = np.concatenate([img, alpha], axis=2)  # BGRA

    h, w = img.shape[:2]
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized  # BGRA

# load assets (put your pngs in same folder)
fruit_images = [
    load_image_rgba("apple.png", 0.15),
    load_image_rgba("banana.png", 0.15),
    load_image_rgba("watermelon.png", 0.15)
]
# keep only loaded ones
fruit_images = [f for f in fruit_images if f is not None]

bomb_image = load_image_rgba("bomb.png", 0.12)

# ------------------------------
# ✋ Mediapipe Hands
# ------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# ------------------------------
# 🎮 Game Variables
# ------------------------------
score = 0
objects = []  # each object: dict with cx, cy, vx, vy (vy negative to go up), image, is_bomb, radius
last_spawn_time = time.time()
game_duration = 30  # seconds
start_time = time.time()

# Camera setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam (VideoCapture failed).")

# ------------------------------
# 🍎 Spawn Object (center coords)
# ------------------------------
def safe_rand_x(width, margin=60):
    if width <= 2 * margin:
        return width // 2
    return random.randint(margin, width - margin)

def spawn_object(width, height):
    cx = safe_rand_x(width, 60)
    cy = height + 50  # start below the frame (we will move it upwards)
    speed_y = random.uniform(8.0, 12.0)
    is_bomb = random.random() < 0.18  # ~18% bombs
    if is_bomb and bomb_image is not None:
        img = bomb_image
    elif fruit_images:
        img = random.choice(fruit_images)
    else:
        img = None

    # Determine radius for collision: if we have an image, use half the max dimension, otherwise fallback
    if img is not None:
        ih, iw = img.shape[:2]
        radius = int(max(iw, ih) * 0.45)
    else:
        radius = 25

    # velocity: moving upward (negative dy)
    return {"cx": cx, "cy": cy, "vy": -speed_y, "image": img, "is_bomb": is_bomb, "radius": radius}

# ------------------------------
# 🖼 Fast alpha overlay using numpy slicing
# ------------------------------
def overlay_png_on_bgr(frame, png_bgra, center_x, center_y):
    """
    Overlay BGRA png_bgra onto frame (BGR) with center at (center_x, center_y).
    Handles clipping automatically.
    """
    if png_bgra is None:
        return

    fh, fw = frame.shape[:2]
    ih, iw = png_bgra.shape[:2]

    # compute top-left of overlay in frame coordinates
    x1 = int(center_x - iw // 2)
    y1 = int(center_y - ih // 2)
    x2 = x1 + iw
    y2 = y1 + ih

    # compute intersection region
    ox1 = max(0, x1)
    oy1 = max(0, y1)
    ox2 = min(fw, x2)
    oy2 = min(fh, y2)

    if ox1 >= ox2 or oy1 >= oy2:
        return  # nothing to draw (fully outside)

    # corresponding region in overlay
    rx1 = ox1 - x1
    ry1 = oy1 - y1
    rx2 = rx1 + (ox2 - ox1)
    ry2 = ry1 + (oy2 - oy1)

    overlay_region = png_bgra[ry1:ry2, rx1:rx2]  # BGRA
    bgr_overlay = overlay_region[..., :3].astype(np.float32)
    alpha = (overlay_region[..., 3:] / 255.0).astype(np.float32)

    frame_region = frame[oy1:oy2, ox1:ox2].astype(np.float32)

    # alpha blending
    blended = alpha * bgr_overlay + (1 - alpha) * frame_region
    frame[oy1:oy2, ox1:ox2] = blended.astype(np.uint8)

# ------------------------------
# 🎮 Main Loop (improved)
# ------------------------------
fps_limit = 30
last_frame_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame read failed, exiting.")
        break

    frame = cv2.flip(frame, 1)
    fh, fw = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # spawn every ~0.8s
    now = time.time()
    if now - last_spawn_time > 0.8:
        obj = spawn_object(fw, fh)
        if obj:
            objects.append(obj)
        last_spawn_time = now

    # move and draw objects
    for obj in objects[:]:
        obj['cy'] += obj['vy']  # moving upward (vy is negative)
        if obj['image'] is not None:
            overlay_png_on_bgr(frame, obj['image'], int(obj['cx']), int(obj['cy']))
        else:
            color = (0, 0, 255) if not obj['is_bomb'] else (0, 0, 0)
            cv2.circle(frame, (int(obj['cx']), int(obj['cy'])), obj['radius'], color, -1)

        # remove if fully off top
        if obj['cy'] < -100:
            objects.remove(obj)

    # hand detection + collision
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            index_finger_tip = hand_landmarks.landmark[8]
            x_index = int(index_finger_tip.x * fw)
            y_index = int(index_finger_tip.y * fh)

            # small white dot for finger tip
            cv2.circle(frame, (x_index, y_index), 8, (255, 255, 255), -1)

            # check collisions with objects (iterate copy)
            for obj in objects[:]:
                dist = math.hypot(obj['cx'] - x_index, obj['cy'] - y_index)
                if dist < obj['radius']:
                    # hit!
                    if obj['is_bomb']:
                        score -= 2
                    else:
                        score += 1
                    play_slice_sound()
                    # remove sliced object
                    if obj in objects:
                        objects.remove(obj)

    # HUD: score + timer
    time_left = max(0, int(game_duration - (time.time() - start_time)))
    cv2.putText(frame, f"Score: {score}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Time: {time_left}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if time_left <= 0:
        cv2.putText(frame, "GAME OVER!", (fw // 2 - 150, fh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.imshow("Fruit Ninja", frame)
        cv2.waitKey(3000)
        break

    cv2.imshow("Fruit Ninja", frame)

    # limit FPS to reduce CPU usage
    elapsed = time.time() - last_frame_time
    sleep_time = max(0, (1.0 / fps_limit) - elapsed)
    if sleep_time > 0:
        time.sleep(sleep_time)
    last_frame_time = time.time()

    # ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
