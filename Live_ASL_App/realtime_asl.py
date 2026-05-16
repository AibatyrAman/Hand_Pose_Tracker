"""
Gerçek zamanlı ASL el alfabesi tanıma.
Çalıştır: venv_mp/bin/python realtime_asl.py

Tuşlar:
  Q → çıkış
  C → ekrandaki harfi onayla, kelimeye ekle
  SPACE → kelimeyi bitir (boşluk ekle)
  BACKSPACE → son harfi sil
"""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import cv2
import numpy as np
import sys
sys.modules['tensorflow'] = None
import mediapipe as mp
import json
from collections import deque

WEIGHTS_PATH = "checkpoints/mlp_weights.npz"
CLASSES_PATH = "checkpoints/landmark_classes.json"
CONFIDENCE_THRESHOLD = 0.75
SMOOTH_WINDOW = 10  # son N karedeki en sık tahmin alınır

# ── Model yükle ─────────────────────────────────────────────────
npz = np.load(WEIGHTS_PATH)

# Dense 1 + BN 1
d1_k  = npz["dense__kernel"];                 d1_b  = npz["dense__bias"]
bn1_g = npz["batch_normalization__gamma"];     bn1_bt = npz["batch_normalization__beta"]
bn1_m = npz["batch_normalization__moving_mean"]; bn1_v = npz["batch_normalization__moving_variance"]
# Dense 2 + BN 2
d2_k  = npz["dense_1__kernel"];               d2_b  = npz["dense_1__bias"]
bn2_g = npz["batch_normalization_1__gamma"];   bn2_bt = npz["batch_normalization_1__beta"]
bn2_m = npz["batch_normalization_1__moving_mean"]; bn2_v = npz["batch_normalization_1__moving_variance"]
# Dense 3
d3_k  = npz["dense_2__kernel"];               d3_b  = npz["dense_2__bias"]
# Dense out
d4_k  = npz["dense_3__kernel"];               d4_b  = npz["dense_3__bias"]

def bn(x, gamma, beta, mean, var, eps=1e-3):
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def predict(features):
    x = features
    x = relu(bn(x @ d1_k + d1_b, bn1_g, bn1_bt, bn1_m, bn1_v))
    x = relu(bn(x @ d2_k + d2_b, bn2_g, bn2_bt, bn2_m, bn2_v))
    x = relu(x @ d3_k + d3_b)
    x = softmax(x @ d4_k + d4_b)
    return x

with open(CLASSES_PATH) as f:
    idx_to_class = json.load(f)

# ── Landmark normalizasyonu ──────────────────────────────────────
def normalize(landmarks):
    arr = np.array([[l.x, l.y, l.z] for l in landmarks], dtype=np.float32)
    arr -= arr[0]
    scale = np.linalg.norm(arr[9])
    if scale > 0:
        arr /= scale
    return arr.flatten()

# ── Arayüz yardımcıları ─────────────────────────────────────────
def draw_rounded_rect(img, x1, y1, x2, y2, color, alpha=0.6):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_hand_skeleton(img, hand_landmarks, h, w):
    mp_drawing.draw_landmarks(img, hand_landmarks,
                              mp_hands.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0,255,180), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(255,255,0), thickness=2))

mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils

# ── Ana döngü ───────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

text_buffer = ""
recent = deque(maxlen=SMOOTH_WINDOW)

print("Başlatıldı. Q=çıkış | C=harfi ekle | SPACE=boşluk | BACKSPACE=sil")

with mp_hands.Hands(max_num_hands=1,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.6) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        label, confidence = "—", 0.0

        if result.multi_hand_landmarks:
            hand_lm = result.multi_hand_landmarks[0]
            draw_hand_skeleton(frame, hand_lm, h, w)

            features = normalize(hand_lm.landmark)
            probs    = predict(features)
            idx      = int(np.argmax(probs))
            confidence = float(probs[idx])
            label    = idx_to_class[str(idx)]
            recent.append(label)
        else:
            recent.append(None)

        # Smoothed prediction (son N karedeki çoğunluk)
        valid = [x for x in recent if x is not None]
        smooth_label = max(set(valid), key=valid.count) if valid else "—"

        # ── Sol panel: tahmin kutusu ─────────────────────────────
        draw_rounded_rect(frame, 10, 10, 300, 130, (20, 20, 20))

        color = (0, 230, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 140, 255)
        cv2.putText(frame, smooth_label, (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.5, color, 6, cv2.LINE_AA)
        cv2.putText(frame, f"{confidence*100:.0f}%", (160, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)
        cv2.putText(frame, "TAHMiN", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)

        # ── Alt panel: yazılan metin ─────────────────────────────
        draw_rounded_rect(frame, 0, h-80, w, h, (20, 20, 20))
        display_text = text_buffer if text_buffer else "C=ekle | SPACE=bosuk | BACKSPACE=sil"
        cv2.putText(frame, display_text, (20, h-28),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("ASL Alphabet — Gercek Zamanli", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and smooth_label not in ("—", "nothing", "del", "space"):
            text_buffer += smooth_label
        elif key == ord(' '):
            text_buffer += " "
        elif key == 8:  # backspace
            text_buffer = text_buffer[:-1]

cap.release()
cv2.destroyAllWindows()
