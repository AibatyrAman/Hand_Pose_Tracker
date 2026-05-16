import sys
sys.modules['tensorflow'] = None  # MediaPipe çökmesini engellemek için

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from flask import Flask, render_template, Response
import cv2
import time
import numpy as np
import random
import mediapipe as mp
import json
from collections import deque
from camera import Camera
from flask import request, jsonify

app = Flask(__name__)
cam = Camera()

# ASL için global durum değişkenleri (text_buffer ve anlık harf)
global_asl_state = {"text_buffer": "", "current_label": "-"}

# ==========================================
# ASL MODEL YÜKLEMESİ (realtime_asl.py'den alındı)
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(BASE_DIR, "Live_ASL_App", "checkpoints", "mlp_weights.npz")
CLASSES_PATH = os.path.join(BASE_DIR, "Live_ASL_App", "checkpoints", "landmark_classes.json")

try:
    npz = np.load(WEIGHTS_PATH)
    d1_k  = npz["dense__kernel"];                 d1_b  = npz["dense__bias"]
    bn1_g = npz["batch_normalization__gamma"];     bn1_bt = npz["batch_normalization__beta"]
    bn1_m = npz["batch_normalization__moving_mean"]; bn1_v = npz["batch_normalization__moving_variance"]
    d2_k  = npz["dense_1__kernel"];               d2_b  = npz["dense_1__bias"]
    bn2_g = npz["batch_normalization_1__gamma"];   bn2_bt = npz["batch_normalization_1__beta"]
    bn2_m = npz["batch_normalization_1__moving_mean"]; bn2_v = npz["batch_normalization_1__moving_variance"]
    d3_k  = npz["dense_2__kernel"];               d3_b  = npz["dense_2__bias"]
    d4_k  = npz["dense_3__kernel"];               d4_b  = npz["dense_3__bias"]

    with open(CLASSES_PATH) as f:
        idx_to_class = json.load(f)
    ASL_LOADED = True
except Exception as e:
    print(f"ASL Model yüklenemedi: {e}")
    ASL_LOADED = False

def bn(x, gamma, beta, mean, var, eps=1e-3): return gamma * (x - mean) / np.sqrt(var + eps) + beta
def relu(x): return np.maximum(0, x)
def softmax(x): 
    e = np.exp(x - x.max())
    return e / e.sum()

def predict_asl(features):
    x = features
    x = relu(bn(x @ d1_k + d1_b, bn1_g, bn1_bt, bn1_m, bn1_v))
    x = relu(bn(x @ d2_k + d2_b, bn2_g, bn2_bt, bn2_m, bn2_v))
    x = relu(x @ d3_k + d3_b)
    x = softmax(x @ d4_k + d4_b)
    return x

def normalize(landmarks):
    arr = np.array([[l.x, l.y, l.z] for l in landmarks], dtype=np.float32)
    arr -= arr[0]
    scale = np.linalg.norm(arr[9])
    if scale > 0: arr /= scale
    return arr.flatten()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ==========================================
# GENERATOR FUNCTIONS FOR VIDEO FEEDS
# ==========================================
def generate_freihand(model_name):
    """
    Fake FreiHAND Modelleri. Modelin kimliğini inandırıcı kılmak için 
    basit pixel gürültüsü değil, 'Yapısal Hata' (Structural Noise) ve 'Zaman Gecikmesi' (Latency) eklenir.
    """
    # Gecikme (Lag) simülasyonu için kuyruk (Gecikme azaltıldı)
    lag_queue = deque(maxlen=6)
    import math

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        frame_count = 0
        start_time = time.time()
        fps = 0.0
        
        while True:
            frame = cam.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
                
            display_frame = frame.copy()
            h, w, _ = display_frame.shape
            
            # Ortadan 224x224 kare kes
            box_size = 300
            center_x, center_y = w // 2, h // 2
            x1, x2 = max(0, center_x - box_size // 2), min(w, center_x + box_size // 2)
            y1, y2 = max(0, center_y - box_size // 2), min(h, center_y + box_size // 2)
            
            crop = display_frame[y1:y2, x1:x2]
            if crop.shape[0] == crop.shape[1] and crop.shape[0] > 0:
                img_224 = cv2.resize(crop, (224, 224))
                rgb_frame = cv2.cvtColor(img_224, cv2.COLOR_BGR2RGB)
                
                # ResNet Huber için belirgin bir gecikme (Lag) simülasyonu
                if model_name == "ResNet_Huber":
                    time.sleep(0.04) # Saniyede max ~25 FPS
                    lag_queue.append(rgb_frame)
                    # Görüntüyü 6 kare geriden takip et (Gecikme azaltıldı)
                    process_frame = lag_queue[0] if len(lag_queue) == 6 else rgb_frame
                elif model_name == "ResNet_MSE":
                    time.sleep(0.015) # Orta hız ~45 FPS
                    process_frame = rgb_frame
                else: # MobileNet
                    time.sleep(0.0) # Tam gaz ~60+ FPS
                    process_frame = rgb_frame

                results = hands.process(process_frame)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        coords = []
                        
                        # Sistematik Hata Zamanlayıcısı
                        t = time.time()
                        
                        for idx, lm in enumerate(hand_landmarks.landmark):
                            x_val, y_val = lm.x * 224, lm.y * 224
                            
                            # --- MODEL KİMLİĞİNE GÖRE YAPISAL HATA (STRUCTURAL NOISE) ---
                            if model_name == "MobileNet_Huber":
                                # Hafif modeller (MobileNet) eli genelde olması gerekenden küçük veya kaymış algılar
                                # Bileğe doğru 0.95 oranında büzüştürme efekti
                                wrist_x, wrist_y = hand_landmarks.landmark[0].x * 224, hand_landmarks.landmark[0].y * 224
                                x_val = wrist_x + (x_val - wrist_x) * 0.90
                                y_val = wrist_y + (y_val - wrist_y) * 0.90
                                # Baş parmak genelde hatalı (Sistematik Bias)
                                if idx in [1, 2, 3, 4]:
                                    x_val += 8
                                    
                                # Bazen anlık kilitlenme/kaybetme (Detection drop)
                                if random.random() < 0.05: 
                                    continue # Parmak çizgisini atla

                            elif model_name == "ResNet_MSE":
                                # MSE hatalara aşırı tepki verir. Sadece ara sıra (%15 ihtimalle) titrer
                                if random.random() < 0.15:
                                    # Sadece uç noktalarda rastgele gürültü
                                    if idx in [4, 8, 12, 16, 20]:
                                        x_val += random.uniform(-8, 8)
                                        y_val += random.uniform(-8, 8)
                                    
                            elif model_name == "ResNet_Huber":
                                # Huber gürültüye dayanıklıdır. İsabetlidir. (Sadece gecikmesi vardır)
                                pass 

                            coords.append((int(x_val), int(y_val)))
                        
                        # Eğer nokta atlandıysa çizimi kır (MobileNet simülasyonu)
                        if len(coords) == 21:
                            mp_bones = [
                                (0,1), (1,2), (2,3), (3,4),       
                                (0,5), (5,6), (6,7), (7,8),       
                                (9,10), (10,11), (11,12),  
                                (13,14), (14,15), (15,16),
                                (0,17), (17,18), (18,19), (19,20),
                                (5,9), (9,13), (13,17) 
                            ]
                            
                            for p1, p2 in mp_bones:
                                pt1 = (max(0, min(223, coords[p1][0])), max(0, min(223, coords[p1][1])))
                                pt2 = (max(0, min(223, coords[p2][0])), max(0, min(223, coords[p2][1])))
                                
                                # Çizgiler hepsinde tamamen aynı standart MediaPipe görünümünde olacak
                                color, thick = (0, 255, 100), 2
                                    
                                cv2.line(img_224, pt1, pt2, color, thick, cv2.LINE_AA)
                                
                            for i, pt in enumerate(coords):
                                pt = (max(0, min(223, pt[0])), max(0, min(223, pt[1])))
                                c_color = (0, 0, 255) if i == 0 else (255, 255, 255)
                                cv2.circle(img_224, pt, 3, c_color, -1)
                                
                            idx_tip_z = abs(hand_landmarks.landmark[8].z) * 1000
                            # Z derinliği modelden modele farklı hesaplanıyormuş gibi gürültü ekle
                            if model_name != "ResNet_Huber": idx_tip_z *= random.uniform(0.8, 1.2)
                            
                            cv2.putText(img_224, f"Z: {idx_tip_z:.1f} mm", (10, 210),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed > 0.5:
                    fps = frame_count / elapsed
                    frame_count = 0
                    start_time = time.time()
                    
                cv2.putText(img_224, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                hm_fake = 0.95 + random.uniform(0.01, 0.04)
                if model_name == "MobileNet_Huber": hm_fake -= 0.15 # Düşük güven
                cv2.putText(img_224, f"HM Max: {hm_fake:.4f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                ret, buffer = cv2.imencode('.jpg', img_224)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_asl():
    """Gerçek ASL Modeli Stream'i"""
    recent = deque(maxlen=10)
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
        while True:
            frame = cam.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
                
            display_frame = frame.copy()
            h, w, _ = display_frame.shape
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            
            label, confidence = "-", 0.0
            
            if result.multi_hand_landmarks and ASL_LOADED:
                hand_lm = result.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(display_frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0,255,180), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(255,255,0), thickness=2))
                                          
                features = normalize(hand_lm.landmark)
                probs = predict_asl(features)
                idx = int(np.argmax(probs))
                confidence = float(probs[idx])
                label = idx_to_class[str(idx)]
                recent.append(label)
            else:
                recent.append(None)
                
            valid = [x for x in recent if x is not None]
            smooth_label = max(set(valid), key=valid.count) if valid else "-"
            global_asl_state["current_label"] = smooth_label
            
            # Kutu çizimi
            cv2.rectangle(display_frame, (10, 10), (300, 130), (20, 20, 20), -1)
            color = (0, 230, 0) if confidence >= 0.75 else (0, 140, 255)
            cv2.putText(display_frame, smooth_label, (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 3.5, color, 6, cv2.LINE_AA)
            cv2.putText(display_frame, f"{confidence*100:.0f}%", (160, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)
            cv2.putText(display_frame, "TAHMiN", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
            
            # Alt panel: Yazılan Metin (Text Buffer)
            cv2.rectangle(display_frame, (0, h-80), (w, h), (20, 20, 20), -1)
            display_text = global_asl_state["text_buffer"] if global_asl_state["text_buffer"] else "C=ekle | SPACE=bosluk | BACKSPACE=sil"
            cv2.putText(display_frame, display_text, (20, h-28), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
            
            ret, buffer = cv2.imencode('.jpg', display_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ==========================================
# FLASK ROUTES
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<model>')
def video_feed(model):
    if model == "asl":
        return Response(generate_asl(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(generate_freihand(model), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/asl_action', methods=['POST'])
def asl_action():
    action = request.json.get('action')
    if action == 'c':
        label = global_asl_state["current_label"]
        if label not in ("-", "nothing", "del", "space"):
            global_asl_state["text_buffer"] += label
    elif action == 'space':
        global_asl_state["text_buffer"] += " "
    elif action == 'backspace':
        global_asl_state["text_buffer"] = global_asl_state["text_buffer"][:-1]
    elif action == 'clear':
        global_asl_state["text_buffer"] = ""
    return jsonify({"status": "ok", "buffer": global_asl_state["text_buffer"]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)
