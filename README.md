# 📌 Vida Sınıflandırma Projesi – Model 1 ve Model 2

Bu repo, iki sınıflı vida görüntü sınıflandırma problemi için hazırlanmıştır:

* **machine_screw** → makine vidası  
* **wood_sheet_metal_screw** → ahşap / sac vidası  

Projede iki farklı model yaklaşımı denenmiştir:

* **Model 1 – Transfer Learning (VGG16)**
* **Model 2 – Sıfırdan Eğitilen Basit CNN (CIFAR-10 tarzı)**

Aşağıda veri seti, ortam ve her iki model için özet bilgiler yer almaktadır.

---

## 1. Veri Seti ve Klasör Yapısı

Ham veri Google Drive üzerinde aşağıdaki yapıdadır:

```text
project-1/
  dataset/
    machine_screw/             # 63 görüntü
    wood_sheet_metal_screw/    # 63 görüntü
````

Model1 ve Model2 için bu veri kontrollü şekilde **train / validation / test** olarak bölünmüştür:

```text
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
* `target_size=(128, 128)` şeklinde yeniden boyutlandırılmıştır (hem Model1 hem Model2).

---

## 2. Ortam ve Kullanılan Kütüphaneler

* **Google Colab**
* **Python 3**
* **Ana kütüphaneler:**

  * TensorFlow / Keras
  * NumPy
  * Matplotlib

Veriye erişim için Google Drive mount edilmiştir:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Ayrıca sonuçların tekrar edilebilir olması için seed ayarı yapılmıştır:

```python
import numpy as np, tensorflow as tf, random

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
```

---

## 3. Model 1 – Transfer Learning ile VGG16 (`Model1.ipynb`)

### 3.1. Temel VGG16 Tabanı

* `VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))`
* Son sınıflandırma bloğu kaldırılmıştır.
* Başlangıç aşamasında:

```python
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

### 3.3. İyileştirilmiş Model 1 – Daha Derin Dense Blok (Final)

Performansı artırmak için üst sınıflandırma bloğu yeniden tasarlanmıştır.

**Final Model Mimarısi (Flatten + Dense):**

* VGG16 (dondurulmuş)
* `Flatten()`
* `Dense(256, activation='relu')`
* `Dropout(0.3)`
* `Dense(128, activation='relu')`
* `Dropout(0.3)`
* `Dense(2, activation='softmax')`

**Eğitim Ayarları:**

* `Adam(1e-4)`
* `categorical_crossentropy`
* `accuracy`
* `EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)`
* En fazla 30 epoch (erken durdurma aktif)

**Performans (özet):**

* Train accuracy ≈ **0.90**
* Validation accuracy ≈ **0.80**
* Test accuracy ≈ **0.61**

> Not: Test seti yalnızca 26 örnek içerdiği için, tek bir görüntünün doğru/yanlış sınıflanması accuracy’yi yaklaşık %3–4 oranında değiştirebilmektedir.

---

### 3.4. Fine-Tuning Denemesi – VGG16 Block 5

Ek deney olarak fine-tuning uygulanmıştır:

* `block5_*` katmanları `trainable = True` yapılmıştır.
* Diğer katmanlar donuk bırakılmıştır.
* Öğrenme oranı düşürülmüştür:

```python
Adam(1e-5)
```

* `EarlyStopping` ile kısa ek eğitim yapılmıştır.

**Sonuç:**

* Validation accuracy yine ≈ **0.80**
* Test accuracy yine ≈ **0.61**

Bu nedenle Fine-Tuning, test performansını anlamlı biçimde artırmadığı için final modele dahil edilmemiş; raporda “ek deney” olarak bırakılmıştır.

---

## 4. Model 2 – Sıfırdan Eğitilen Basit CNN (`model2.ipynb`)

Model 2’de amaç, **transfer learning kullanmadan**, CIFAR-10 benzeri **basit bir CNN mimarisini sıfırdan** eğitip aynı veri seti üzerinde performansı gözlemlemektir. Böylece Model 1 ve Model 2 sonuçları doğrudan karşılaştırılabilir.

### 4.1. Veri ve Girdi Ayarları

Model 2 de aynı `dataset_split` yapısını kullanır:

* Train: 80 görüntü (40 + 40)
* Validation: 20 görüntü (10 + 10)
* Test: 26 görüntü (13 + 13)

Tüm görüntüler:

```python
ImageDataGenerator(rescale=1./255)
target_size = (128, 128)
batch_size = 8  # veya 16
class_mode = 'categorical'
```

şeklinde Keras `flow_from_directory` ile okunmuştur. Train/val/test için ayrı generator’lar tanımlanmıştır.

---

### 4.2. Model 2 CNN Mimarisi

Model 2, üç konvolüsyon bloğu ve ardından basit bir tam bağlı kısımdan oluşan klasik bir CNN’dir.

```python
model2 = Sequential([
    # Giriş
    Input(shape=(128, 128, 3)),

    # Blok 1
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2)),

    # Blok 2
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2)),

    # Blok 3
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2)),

    # Tam bağlı kısım
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])
```

Bu yapı, CIFAR-10 örneklerinde kullanılan basit CNN’lere benzer olacak şekilde tasarlanmıştır; herhangi bir ön-eğitim (pretrained weights) kullanılmamıştır.

---

### 4.3. Eğitim Ayarları

Model 2 için kullanılan temel eğitim ayarları:

```python
model2.compile(
    optimizer=Adam(learning_rate=2e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop2 = EarlyStopping(
    monitor='val_accuracy',
    patience=8,
    restore_best_weights=True
)

EPOCHS = 50  # early stopping ile genelde daha erken duruyor
```

Eğitim sırasında:

* Eğitim ve doğrulama doğruluk/kayıp değerleri kayıt altına alınmış,
* `Model 2 - Eğitim ve Doğrulama Doğruluğu` ve
  `Model 2 - Eğitim ve Doğrulama Kayıp Değerleri` grafikleri çizdirilmiştir.

---

### 4.4. Model 2 Sonuçları (Özet)

Seçilen final konfigürasyon için gözlenen tipik değerler:

* **En yüksek validation accuracy** ≈ **0.65–0.70**
* Eğitim doğruluğu epoch sonlarında ≈ **0.70** seviyesine yaklaşmaktadır.
* **Test accuracy** ≈ **0.42**
  (26 örnek için bu, yaklaşık 11/26 doğru sınıflama anlamına gelir.)

Loss grafikleri incelendiğinde:

* Hem train loss hem val loss zamanla azalmakta,
* Aralarındaki fark çok açılmadığı için aşırı overfitting gözlenmemektedir,
* Ancak küçük veri seti ve sıfırdan eğitim nedeniyle modelin genelleme kapasitesi sınırlı kalmaktadır.

Bu sonuçlar, sıfırdan eğitilen basit CNN’in bu veri setinde **orta düzey bir performans** sağladığını, ancak transfer learning’e göre daha zayıf kaldığını göstermektedir.

---

## 5. Model 1 ve Model 2 Karşılaştırması

Aynı veri bölünmesi üzerinde iki modelin test performansları kabaca şöyledir:

* **Model 1 (VGG16, transfer learning)**

  * Test accuracy ≈ **0.61**
* **Model 2 (basit CNN, sıfırdan eğitim)**

  * Test accuracy ≈ **0.42**

Bu farkın başlıca nedenleri:

1. **Önceden Eğitilmiş Özellikler:**
   VGG16, ImageNet üzerinde eğitildiği için kenar, doku, şekil gibi düşük/orta seviye görsel özellikleri zaten iyi öğrenmiş durumdadır. Küçük vida veri seti üzerinde sadece üst sınıflandırıcı katmanların eğitilmesi bile yüksek performans sağlamaktadır.

2. **Veri Miktarı ve Sıfırdan Eğitim:**
   Model 2’de tüm ağırlıklar sıfırdan rastgele başlatılmıştır. Her sınıf için yalnızca 40 eğitim görüntüsü (toplam 80 örnek) ile bu ağın hem düşük seviyeli hem yüksek seviyeli özellikleri aynı anda öğrenmesi zordur. Bu nedenle test setinde genelleme performansı sınırlı kalmaktadır.

Sonuç olarak:

> Küçük ve sınırlı bir veri setinde, **transfer learning (Model 1)** yaklaşımı, **sıfırdan eğitilen basit CNN (Model 2)** yaklaşımına göre daha yüksek ve daha kararlı bir performans sunmuştur.

---

## 6. Çalıştırma Adımları (Kısa Özet)

1. **Drive bağlantısı**

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Veri hazırlama**

   * Ham veri: `dataset/`
   * Train/Val/Test: `dataset_split/` içinde

     * Train: 80, Val: 20, Test: 26 örnek.

3. **Model 1 (VGG16 – Transfer Learning) – `Model1.ipynb`**

   * VGG16 tabanını yükle (`include_top=False`, `weights='imagenet'`).
   * Üst sınıflandırıcı bloğu (Flatten + Dense(256,128) + Dropout) ekle.
   * `Adam(1e-4)` ile eğit, EarlyStopping uygula.
   * Eğitim/val grafiklerini çiz ve test doğruluğunu raporla.

4. **Model 2 (Basit CNN) – `model2.ipynb`**

   * Aynı `dataset_split` klasörünü kullan.
   * 3 konvolüsyon bloğu + Flatten + Dense(256) + Dropout + Dense(2) mimarisi kur.
   * `Adam(2e-4)` ile eğit, EarlyStopping uygula.
   * Eğitim/val grafiklerini çiz ve test doğruluğunu raporla.

5. **Karşılaştırma**

   * Model1 ve Model2’nin validation/test doğruluklarını karşılaştır.
   * Transfer learning’in küçük veri setlerinde sağladığı avantajı tartış.

