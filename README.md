# 📌 Vida Sınıflandırma Projesi – Model 1 (Transfer Learning, VGG16)

Bu repo, iki sınıflı vida görüntü sınıflandırma problemi için hazırlanmıştır:

* **machine_screw** → makine vidası
* **wood_sheet_metal_screw** → ahşap / sac vidası

Bu README, özellikle **Model1.ipynb** dosyasında yapılan **transfer learning (VGG16)** deneylerini özetler.

---

## 1. Veri Seti ve Klasör Yapısı

Ham veri Google Drive üzerinde aşağıdaki yapıdadır:

```
project-1/
  dataset/
    machine_screw/             # 63 görüntü
    wood_sheet_metal_screw/    # 63 görüntü
```

Model1 için bu veri kontrollü şekilde **train / validation / test** olarak bölünmüştür:

```
project-1/
  dataset_split/
    train/
      machine_screw/           # 40 görüntü
      wood_sheet_metal_screw/  # 40 görüntü
    val/
      machine_screw/           # 10 görüntü
      wood_sheet_metal_screw/  # 10 görüntü
    test/
      machine_screw/           # 13 görüntü
      wood_sheet_metal_screw/  # 13 görüntü
```

* Toplam görüntü: **126**
* Train: **80**
* Validation: **20**
* Test: **26**

Eğitim sırasında tüm görüntüler:

* `rescale=1./255` ile normalize edilmiştir,
* `target_size=(128, 128)` şeklinde yeniden boyutlandırılmıştır.

---

## 2. Ortam ve Kullanılan Kütüphaneler

* **Google Colab**
* **Python 3**
* **Ana kütüphaneler:**

  * TensorFlow / Keras
  * Matplotlib

Veriye erişim için Google Drive mount edilmiştir.

---

## 3. Model 1 – Transfer Learning ile VGG16

### 3.1. Temel VGG16 Tabanı

* `VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))`
* Son sınıflandırma bloğu kaldırılmıştır.
* Başlangıç aşamasında:

  ```
  base_model.trainable = False
  ```

---

### 3.2. İlk Model1 Denemesi – GAP + Dense

Model mimarisi:

* VGG16 (dondurulmuş)
* `GlobalAveragePooling2D`
* `Dense(128, activation='relu')`
* `Dropout(0.5)`
* `Dense(2, activation='softmax')`

**Eğitim Ayarları:**

* `Adam(1e-4)`
* `categorical_crossentropy`
* `accuracy` metriği
* `EPOCHS = 30`
* `EarlyStopping(patience=3)`

**Sonuç (özet):**

* Train accuracy ≈ **0.90**
* Validation accuracy ≈ **0.70–0.75**
* Test accuracy ≈ **0.50**

Bu nedenle geliştirme ihtiyacı görülmüştür.

---

## 4. İyileştirilmiş Model 1 – Daha Derin Dense Blok

Performansı artırmak için üst sınıflandırma bloğu yeniden tasarlanmıştır.

### 4.1. Final Model Mimarısi (Flatten + Dense)

* VGG16 (dondurulmuş)
* `Flatten()`
* `Dense(256, activation='relu')`
* `Dropout(0.3)`
* `Dense(128, activation='relu')`
* `Dropout(0.3)`
* `Dense(2, activation='softmax')`

**Eğitim Ayarları:**

* `Adam(1e-4)`
* `EarlyStopping(patience=5)`
* 30 epoch (erken durdurma aktif)

**Performans (özet):**

* Train accuracy ≈ **0.90**
* Validation accuracy ≈ **0.80**
* Test accuracy ≈ **0.61**
  (26 örnek olduğu için ±1 görüntü %3–8 arasında değişim yapabiliyor.)

---

## 5. Fine-Tuning Denemesi – VGG16 Block 5

Ek deney olarak fine-tuning uygulanmıştır:

* `block5_*` katmanları `trainable = True` yapılmıştır.
* Öğrenme oranı:

  ```
  Adam(1e-5)
  ```
* EarlyStopping ile kısa ek eğitim yapılmıştır.

**Sonuç:**

* Validation accuracy yine ≈ **0.80**
* Test accuracy yine ≈ **0.61**

> Bu nedenle Fine-Tuning, test performansını anlamlı biçimde artırmadığı için final modele dahil edilmemiştir.

---

## 6. Çalıştırma Adımları (Kısa Özet)

1. **Drive bağlantısı**

   ```
   drive.mount('/content/drive')
   ```

2. **Veri hazırlama**

   * Ham veri → eğitim/validation/test klasörlerine bölünür.
   * Train: 80, Val: 20, Test: 26 örnek.

3. **ImageDataGenerator ayarları**

   * `128×128`, `rescale=1/255`

4. **Model 1 eğitimi (final mimari)**

   * Flatten + Dense(256,128) + Dropout
   * EarlyStopping
   * Test doğruluğu hesaplanır.

5. **Ek denemeler**

   * GAP tabanlı model
   * Block5 Fine-Tuning

