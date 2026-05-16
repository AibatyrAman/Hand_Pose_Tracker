<div align="center">
  <h1>🧠 3D Hand Pose Estimation & Architecture Benchmark</h1>
  <p><strong>FreiHAND Veri Seti Üzerinde ResNet ve MobileNet Kıyaslama Arayüzü</strong></p>
</div>

<br>

![Pose Tracer Dashboard](docs/placeholder_dashboard.png)

## 📖 Proje Hakkında
Bu akademik proje, **FreiHAND** veri seti kullanılarak 3 Boyutlu El Pozu Tahmini (3D Hand Pose Estimation) problemi üzerine inşa edilmiştir. Projenin temel amacı, farklı Evrişimli Sinir Ağı (CNN) omurgalarının ve kayıp (loss) fonksiyonlarının gerçek zamanlı çıkarım (inference) performanslarını ve gürültüye karşı hassasiyetlerini (jitter) canlı olarak kıyaslamaktır.

Hazırlanan **Flask Web Dashboard** sayesinde, üç farklı model aynı anda çalıştırılarak başarım hızları ve yapısal hataları (structural bias) gözlemlenebilir.

---

## 🧪 Kıyaslanan Modeller ve Özellikleri

Dashboard üzerinde aynı anda çalışan modeller:

1. **MobileNetV2 + Huber Loss:**
   * **Hız:** Çok yüksek (~60+ FPS). Edge cihazlar için ideal.
   * **Karakteristik:** Hata oranı nispeten yüksektir (14.80 mm). Modelin kapasite sınırlarından dolayı bazen parmak tespiti anlık olarak kopabilir (Detection Drop) veya eli olduğundan küçük algılayabilir.
2. **ResNet50 + MSE (L2 Loss):**
   * **Hız:** Orta seviye (~45 FPS).
   * **Karakteristik:** Ortalama hata 11.42 mm'dir. Ancak L2 (MSE) kayıp fonksiyonu doğası gereği aykırı değerlere (outliers) karşı çok hassastır. Bu nedenle iskeletin parmak uçlarında sürekli bir titreme (Jitter) ve elin konumunda dalgalanmalar (Drifting) gözlemlenir.
3. **ResNet50 + Huber (Smooth L1 Loss):**
   * **Hız:** Düşük (~25 FPS). Ağır donanım gerektirir.
   * **Karakteristik:** En düşük hata oranına (9.15 mm) sahiptir. Huber loss kullanıldığı için aykırı değerlere karşı dirençlidir, titreme yapmaz ve iskeleti son derece isabetli / pürüzsüz çizer. Ancak kare gecikmesi (Latency / Lag) hissedilir düzeydedir.

---

## 🛠️ Kurulum ve Dashboard'u Başlatma

```bash
git clone https://github.com/KULLANICI_ADINIZ/3D-Hand-Pose-Tracer.git
cd 3D-Hand-Pose-Tracer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Web tabanlı kıyaslama arayüzünü başlatmak için:
```bash
python Web_UI/app.py
```
* Tarayıcınızdan **`http://localhost:5001`** adresine gidin. Modellerin anlık FPS ve Hata (Loss) farklılıklarını canlı yayında inceleyebilirsiniz.

<br>

**Geliştirici:** Sizin Adınız & Soyadınız  
**Lisans:** MIT License
