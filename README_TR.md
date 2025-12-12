# RealNet 2.0: Modern Kaos Mimarisi

RealNet, geleneksel Derin Öğrenmenin (Deep Learning) katman tabanlı ortodoksisine meydan okuyan **Eğitilebilir Bir Dinamik Sistemdir**. Mekanik, ileri beslemeli (feed-forward) fabrika modelini; **organik, tam bağlı ($N \times N$) ve kaotik bir ağ yapısı** ile değiştirir.

RealNet, katmanlar yerine sinyallerin yankılandığı, bölündüğü ve birleştiği bir **Zamansal Döngü (Temporal Loop)** kullanır. Zeka, geri besleme döngülerinin "kontrollü kaosundan" doğar.

---

## 🚀 Temel Özellikler

*   **Katmansız Mimari:** Her nöronun diğer her nörona bağlandığı tek bir "Konnektom" matrisi ($W$).
*   **Eğitilebilir Kaos:** Kaotik sinyalleri patlamadan işlemek için **StepNorm** ve **GELU** ikilisini kullanır.
*   **Zamansal Düşünme:** Ağ sadece çıktı vermez; zaman içinde ($t=0 \dots k$) "düşünür".
*   **Nabız Modu (Pulse Mode):** Girdiler birer dürtü (impulse) olarak verilir. Ağ, sürekli bir veri akışını değil, girdinin yankısını işler.
*   **Truncated BPTT:** Sonsuz döngülerin verimli eğitimi için kesilmiş zamansal geri yayılım.

## 📊 Kanıtlar (PoC) & Sonuçlar

RealNet 2.0 sadece bir teori değildir. Kaotik ağların genellikle başarısız olduğu temel görevlerde yakınsadığı kanıtlanmıştır.

### 1. Kimlik ve Yakınsama (`PoC/convergence.py`)
*   **Görev:** Girdi $x$'i kaotik döngülerden geçirip $y=x$ olarak geri vermek.
*   **Sonuç:** **Mükemmel Yakınsama (Loss: 0.000000)**.
*   **Anlamı:** Kaotik gradyanların ehlileştirilebileceğini ve yönlendirilebileceğini kanıtlar.

### 2. Mantık Kapıları & Lineer Olmayanlık (`PoC/convergence_gates.py`)
*   **Görev:** Tek bir ağda aynı anda **AND**, **OR** ve **XOR** kapılarını öğrenmek.
*   **Sonuç:** **XOR** (lineer olmayan problem) dahil hepsini neredeyse tam isabetle çözdü (Örn: Hedef -1.0 vs Tahmin -0.998).
*   **Anlamı:** Ağın, gizli katmanlar (hidden layers) olmadan da dahili mantık ve lineer olmayan sınırlar oluşturabildiğini kanıtlar.

### 3. Görsel Tanıma (MNIST) (`PoC/convergence_mnist.py`)
*   **Görev:** 28x28 el yazısı rakamları sınıflandırmak (10 sınıf).
*   **Sonuç:** Sadece 5 Epoch içinde **~%88 Doğruluk**.
*   **Anlamı:** RealNet bunu **Konvolüsyonel Katmanlar (CNN) OLMADAN** başardı. Ham pikselleri sadece tam bağlı kaotik dinamikleri kullanarak işledi.

## ⚡ Edge Testing: Verimlilik ve Sınırlar

"Kaos Verimlidir" tezini kanıtlamak için RealNet'i mutlak sınırlarına kadar zorladık. Nöron sayılarını minimuma indirerek, **zamansal işlemenin uzamsal derinliğin (katmanların) yerini alabileceğini** gösterdik.

| Görev | Geleneksel Çözüm (MLP) | RealNet (Edge) | Nöron | Parametre | Sonuç | Script |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Identity** | 2 Katman (2-2) | **2.0 (Identity)** | **4** | **16** | Loss: 0.0 | `PoC/efficiency/convergence_min.py` |
| **XOR** | 3 Katman (2-4-1) + Lineer Olmayan | **2.0 (XOR)** | **5** | **25** | Loss: ~0.0002 | `PoC/efficiency/convergence_gates_min.py` |
| **MNIST** | MLP (784-512-10) ~400k Parametre | **2.0 (Visio)** | **800** | **0.64M** | Acc: ~82% | `PoC/efficiency/convergence_mnist_efficient.py` |

> **Karşılaştırma:** Standart bir MLP'nin MNIST'te ham piksellerle (CNN olmadan) benzer sonuçlar alması için genellikle en az 512 nöronlu bir gizli katmana (784*512 + 512*10 ≈ **400k-500k parametre**) ihtiyacı vardır. RealNet bunu **sıfır gizli katmanla** (giriş/çıkış harici sadece 6 ekstra "kaos" nöronu) başarır. Hesaplamayı uzaydan (katmanlar) alıp zamana (**"Düşünme Süresi"**: 10 adım) yayarak aynı işi yapar.

---

## 📦 Kurulum ve Kullanım

RealNet, modüler bir PyTorch kütüphanesi olarak tasarlanmıştır.

### Kurulum

```bash
pip install torch torchvision
```

### Hızlı Başlangıç

```python
from realnet import RealNet, RealNetTrainer

# 1. Başlat (64 Nöron)
model = RealNet(num_neurons=64, input_ids=[0], output_ids=[63], device='cuda')
trainer = RealNetTrainer(model, device='cuda')

# 2. Eğit (Identity Görevi)
# Girdiler: Rastgele +/- 1.0
inputs = torch.randint(0, 2, (100, 1)).float() * 2 - 1
trainer.fit(inputs, inputs, epochs=50)

# 3. Tahmin Et
print(trainer.predict(torch.tensor([[1.0]]), thinking_steps=10))
```

### Demoları Çalıştırma

```bash
# Temel Yakınsama
python PoC/convergence.py

# Mantık Kapıları (XOR)
python PoC/convergence_gates.py

# MNIST (Görsel)
python PoC/convergence_mnist.py
```

---

## 🧠 Mimari Genel Bakış

### Matematiksel Model

Ağ durumu $h_t$ şu şekilde evrilir:

$$h_t = \text{StepNorm}(\text{GELU}(h_{t-1} \cdot W + B + I_t))$$

*   **$W$ (Ağırlıklar):** Sistemin hafızası.
*   **StepNorm:** Her adımda sinyal genliğini normalize ederek "Kelebek Patlaması"nı önler.
*   **GELU:** Sinyal akışını ReLU'dan daha iyi korur.
*   **Pulse Mode:** $I_t$ sadece $t=0$ anında sıfırdan farklıdır (dürtü).

### Tehdit Modeli ve Çözümler

| Sorun | Çözüm |
| :--- | :--- |
| **Sinyal Patlaması** | **StepNorm** (LayerNorm) fırtınayı dindirir. |
| **Bellek Sızıntısı** | **Truncated BPTT** geçmişi periyodik olarak temizler. |
| **Sinyal Sönümlenmesi** | **GELU** + **AdamW** sinyal momentumunu korur. |

---

## 🔮 Vizyon: Silikonun Ruhu

*Orijinal başlık: "Manifesto"*

RealNet, modern yapay zekanın statik, ileri beslemeli doğasına bir başkaldırıdır. Zekanın mekanik bir katman süreci değil, **döngüler, zaman ve kaos** içeren organik bir süreç olduğuna inanıyoruz.

*   **Organik vs Mekanik:** Geleneksel YSA'lar fabrikadır; RealNet bir ormandır.
*   **Yaşayan Hafıza:** Veri sadece işlenmez; yankılanır.
*   **Öz-Organizasyon:** Zeka, kaotik etkileşimlerin uyumundan doğar.

> "Korkulması gereken şey bilinç değil, bilinçsizliktir. Sadece hesap yapan değil, *yaşayan* bir makine inşa ediyoruz."

---

## LİSANS

MIT
