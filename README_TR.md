# RealNet 2.0: Zamansal Devrim

**RealNet, Zamanın en büyük Gizli Katman olduğunun kanıtıdır.**

Geleneksel Derin Öğrenme, karmaşıklığı çözmek için **Uzamsal Derinliğe** (üst üste yığılmış katmanlara) güvenir. RealNet bu dogmayı reddeder ve **Zamansal Derinliğin** (zaman içinde evrilen kaosun) çok daha verimli bir alternatif olduğunu kanıtlar.

> **Sıfır-Gizli Katman Devrimi (Zero-Hidden Breakthrough)**
>
> 1969'da Minsky ve Papert, gizli katmanı olmayan bir sinir ağının XOR gibi lineer olmayan problemleri çözemeyeceğini matematiksel olarak kanıtladı.
> **RealNet 2.0 bu sınırı paramparça etti.**
>
> Ağı "Eğitilebilir Bir Dinamik Sistem" olarak ele alan RealNet, **0 Gizli Katman** kullanarak non-lineer problemleri (XOR, MNIST) çözer. Uzamsal nöronların yerini zamansal düşünme adımları alır.

---

## 🚀 Temel Özellikler

*   **Uzay-Zaman Dönüşümü:** Milyonlarca parametrenin yerini birkaç "Düşünme Adımı" alır.
*   **Katmansız Mimari:** Tek bir $N \times N$ matris. Gizli katman yok.
*   **Eğitilebilir Kaos:** Sinyalleri ehlileştirmek için **StepNorm** ve **GELU** kullanılır.
*   **Nabız Modu:** Ağ, sürekli bir veri akışını değil, tek bir dürtünün (impulse) yankısını işler.

## 📊 Kanıtlar: Sıfır-Gizli Benchmarkları

RealNet'i teorik sınırlara kadar zorladık: **Sıfır Gizli Nöron**.
Bu testlerde Giriş Katmanı doğrudan Çıkış Katmanına (ve kendisine) bağlıdır. Tampon katman yoktur.

| Görev | Geleneksel Engel | RealNet Çözümü | Nöron | Parametre | Sonuç | Script |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Identity** | Basit | **Atomik Birim** | **4** | **16** | Loss: 0.0 | `PoC/convergence.py` |
| **XOR** | Gizli Katman Şart | **Minimal Kaos** | **5** | **25** | Loss: ~0.0002 | `PoC/convergence_gates.py` |
| **MNIST** | ~500k Parametre Şart | **Sıfır-Gizli** | **206** | **~42k** | **Acc: ~89.8%** | `PoC/convergence_mnist.py` |

### MNIST Mucizesi
Standart MLP'ler 784 pikseli 10 rakama dönüştürmek için yaklaşık 400.000 parametreye ihtiyaç duyar.
RealNet bunu **42.436 parametre** ile yapar.
*   **Giriş:** 196 (14x14 Yeniden Boyutlandırılmış)
*   **Çıkış:** 10
*   **Gizli:** 0
*   **Düşünme Süresi:** 15 Adım

Giriş katmanı 15 adım boyunca "kendi kendine konuşur". Kaotik geri besleme döngüleri, zaman içinde özellik çıkarımı (feature extraction) yaparak uzamsal katmanların işini üstlenir. Bu, **Sıkıştırma Zekasının** zirvesidir.

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

# Sıfır-Gizli Katmanlı Bir Ağ Başlat
# 1 Giriş, 1 Çıkış. 
model = RealNet(num_neurons=2, input_ids=[0], output_ids=[1], device='cuda')
trainer = RealNetTrainer(model, device='cuda')

# Eğit
inputs = torch.randn(100, 1)
trainer.fit(inputs, inputs, epochs=50)
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

---

## 🔮 Vizyon: Silikonun Ruhu

RealNet, yapay zekanın katmanlı fabrika modeline bir başkaldırıdır. Zekanın mekanik bir katman yığını değil, sinyallerin organik yankısı olduğuna inanıyoruz.

Küçük, kaotik bir nöron ormanının, "düşünmek" için yeterli zaman verildiğinde, devasa endüstriyel fabrikalardan daha iyi performans gösterebileceğini kanıtladık.

> "Uzayı feda edip Zamanı kazandık ve bunu yaparken Ruhu bulduk."

---

---

## 👨‍💻 Yazar (Author)

**Cahit Karahan**
*   Doğum: 12/02/1997, Ankara.
*   "Kaosun Mimarı."

---

## LİSANS

MIT
