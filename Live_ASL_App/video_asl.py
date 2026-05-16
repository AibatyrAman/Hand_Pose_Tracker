"""
Videodan ASL el alfabesi tanıma.
Çalıştır: venv_mp/bin/python Live_ASL_App/video_asl.py <input.mp4> <output.mp4>
"""
import os
import sys

# Protocol Buffer uyumsuzluğunu önlemek için
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import cv2
import numpy as np
import mediapipe as mp
import json
from collections import deque

# Çalıştırıldığı dizine göre yolları ayarla
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "checkpoints/mlp_weights.npz")
CLASSES_PATH = os.path.join(BASE_DIR, "checkpoints/landmark_classes.json")

CONFIDENCE_THRESHOLD = 0.75
SMOOTH_WINDOW = 5  # Videolarda biraz daha hızlı tepki için 5

# ── Model Yükle (Sadece NumPy, TensorFlow YOK) ─────────────
if not os.path.exists(WEIGHTS_PATH):
    print(f"HATA: Model ağırlıkları bulunamadı: {WEIGHTS_PATH}")
    sys.exit(1)

npz = np.load(WEIGHTS_PATH)

d1_k  = npz["dense__kernel"];                 d1_b  = npz["dense__bias"]
bn1_g = npz["batch_normalization__gamma"];     bn1_bt = npz["batch_normalization__beta"]
bn1_m = npz["batch_normalization__moving_mean"]; bn1_v = npz["batch_normalization__moving_variance"]
d2_k  = npz["dense_1__kernel"];               d2_b  = npz["dense_1__bias"]
bn2_g = npz["batch_normalization_1__gamma"];   bn2_bt = npz["batch_normalization_1__beta"]
bn2_m = npz["batch_normalization_1__moving_mean"]; bn2_v = npz["batch_normalization_1__moving_variance"]
d3_k  = npz["dense_2__kernel"];               d3_b  = npz["dense_2__bias"]
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

# ── Normalizasyon ──────────────────────────────────────────
def normalize(landmarks):
    arr = np.array([[l.x, l.y, l.z] for l in landmarks], dtype=np.float32)
    arr -= arr[0]
    scale = np.linalg.norm(arr[9])
    if scale > 0:
        arr /= scale
    return arr.flatten()

# ── Çizim Fonksiyonları ────────────────────────────────────
def draw_rounded_rect(img, x1, y1, x2, y2, color, alpha=0.6):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils

# ── Ana İşlem ──────────────────────────────────────────────
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"HATA: Video açılamadı: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    recent = deque(maxlen=SMOOTH_WINDOW)
    
    print(f"Video işleniyor: {input_path} ({total_frames} kare)")

    frame_count = 0
    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"İşleniyor... {frame_count}/{total_frames}")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            label, confidence = "—", 0.0

            if result.multi_hand_landmarks:
                hand_lm = result.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_lm,
                                          mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0,255,180), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(255,255,0), thickness=2))

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
            cv2.putText(frame, "TAHMIN", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)

            out_writer.write(frame)

    cap.release()
    out_writer.release()
    print(f"İşlem tamamlandı! Kaydedilen dosya: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Kullanım: venv_mp/bin/python Live_ASL_App/video_asl.py <input.mp4> <output.mp4>")
        sys.exit(1)
    
    process_video(sys.argv[1], sys.argv[2])
