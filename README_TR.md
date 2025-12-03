# RealNet: Beyin İlhamlı Katmansız Sinir Mimarisi

![License](https://img.shields.io/badge/license-MIT-blue.svg)

[🇺🇸 Read in English](./README.md) | [📜 Orijinal Manifesto (Eski Metin)](./MANIFESTO.md)

## Özet

RealNet, geleneksel katmanlı mimarilerden (FNN'ler, CNN'ler, Transformer'lar) temelden ayrılan, yapay sinir ağlarında yeni bir paradigma sunar. Biyolojik beynin kaotik ancak verimli bağlantısallığından ilham alan RealNet, her nöronun diğer her nöronla bağlantı kurma potansiyeline sahip olduğu, tamamen birbirine bağlı, katmansız bir topoloji kullanır. Bu mimari, çok boyutlu veri iletimini (2D'den 5D+'ya) kolaylaştırır ve dairesel veri döngüleri aracılığıyla kısa süreli hafızanın, benzersiz bir "İleri Ateşle, İleri Bağla" (Fire Forward, Wire Forward - FFWF) öğrenme algoritması aracılığıyla ise uzun süreli hafızanın ortaya çıkmasını sağlar. RealNet, statik veri setlerinin kısıtlamaları olmaksızın aktif öğrenme, rüya görme ve öz-düzenleme için teorik yetenekler sergiler.

## 1. Giriş

Derin öğrenmedeki hakim yaklaşım, yapılandırılmış katmanlara ve geri yayılıma (backpropagation) dayanır. Etkili olsalar da, bu yöntemler genellikle doğal sinir sistemlerinin dinamik uyarlanabilirliğinden ve biyolojik makullüğünden yoksundur. RealNet, beynin "nöron çorbası" yaklaşımını taklit ederek bu sınırlamaları ele alır.

RealNet'te "katman" kavramı ortadan kaldırılmıştır. Ağ, yönlülüğün dayatılmak yerine kendiliğinden ortaya çıktığı kaotik bir bağlantı ağıdır. Bu yapı şunlara olanak tanır:
*   **Dinamik Topoloji:** Ağ, etkin yapısını veri akışına göre uyarlayabilir.
*   **Zamansal İşleme:** Bilgi, karmaşık zamansal bağımlılıklara izin verecek şekilde sürekli zaman adımlarında işlenir.
*   **Öz-Düzenleme:** Ağ, Hebbian öğrenmeye benzer ancak zamansal uygulamasında farklı olan aktivite korelasyonlarına dayanarak kendi bağlantılarını iyileştirir.

## 2. Teorik Mimari

### 2.1. Topoloji ve Bağlantısallık
Ağ, bir nöron ve bağlantı koleksiyonundan oluşur. Bağlantıların yalnızca bitişik katmanlar arasında olduğu FNN'lerin aksine, bir RealNet nöronu sistemdeki diğer herhangi bir nörondan girdi alabilir ve ona çıktı gönderebilir.
*   **Kaotik Bağlantısallık:** Bu, karmaşık, tekrarlayan (recurrent) yapıların oluşumuna izin verir.
*   **Dairesel Döngüler (Kısa Süreli Hafıza):** Veriler geri besleme döngülerinde sıkışıp kalabilir, bu da etkili bir şekilde kısa süreli bir hafıza tamponu görevi görür. Bu döngüler periyodik sinyaller yayarak ağın durumunu birden fazla zaman adımında etkiler.
*   **Boyutluluk:** Bağlantı modeli, geleneksel katmanların düz temsillerini aşarak keyfi boyutlarda veri temsilini destekler.

### 2.2. Nöron Dinamikleri
Her nöron, zamanla gelişen bir iç durum (state) tutar.
*   **Birikmiş İstatistikler:** Nöronlar ortalama, maksimum ve minimum ateşleme değerlerini takip eder.
*   **Uyarlanabilir Duyarlılık:** Bu istatistikler, aktivasyon fonksiyonunu dinamik olarak ölçeklendirmek için kullanılır, böylece nöronun tekrarlayan arka plan gürültüsüne alışırken yeni uyaranlara karşı duyarlı kalması sağlanır.

### 2.3. Aktivasyon Fonksiyonu: Uyarlanabilir Tanh
RealNet, ağ sinyallerinin dinamik aralığını işlemek için tasarlanmış özel, uyarlanabilir bir aktivasyon fonksiyonu kullanır. Doygunluğu önlemek ve verimli gradyan akışını (kavramsal olarak) sağlamak için dinamik ölçeklendirme ve normalizasyon içerir.

**Matematiksel Formülasyon:**

$$y = \frac{\tanh\left( k \cdot \frac{x - x_{ort}}{ \frac{x_{max} - x_{min}}{2} + \frac{x_{max} + x_{min} - 2x_{ort}}{2} \cdot \text{sgn}(x - x_{ort}) } \right)}{\tanh(k)}$$

Burada:
*   $x$ girdi değeridir.
*   $x_{ort}, x_{max}, x_{min}$ nöronun çalışan istatistikleridir.
*   $k$ bir sabittir (Altın Oran $\phi \approx 1.618$ veya $3$ önerilir).
*   $\text{sgn}(z) = \frac{z}{|z| + \epsilon}$ türevlenebilir bir işaret fonksiyonudur.

**Mekanizma:**
1.  **Dinamik Ölçeklendirme:** Payda, $x$'in ortalamanın üzerinde veya altında olmasına bağlı olarak ayarlanır ve girdiyi nöronun tarihsel aralığına göre etkili bir şekilde normalleştirir.
2.  **Normalizasyon:** $\tanh(k)$ ile bölme, girdiler tarihsel uç noktalara çarpsa bile çıktı aralığının kesinlikle $[-1, 1]$ olmasını sağlar.
3.  **Alışma (Habituation):** Bir nöron tutarlı bir şekilde ateşlendiğinde, $x_{ort}$ kayar ve fonksiyon nöronu o kararlı duruma karşı duyarsızlaştırır, tekrarlayan gürültüyü filtreler ve anormallikleri (sıçramaları) vurgular.

## 3. Algoritmik Çekirdek

### 3.1. Çıkarım Motoru (Inference Engine)
RealNet'te çıkarım, tek geçişli bir yayılım değil, zaman adımlı bir süreçtir.
1.  **Biriktirme:** Bağlantılar, önceki zaman adımından tamponlanmış değerleri hedef nöronlara iletir.
2.  **Durum Güncellemesi:** (İsteğe bağlı) Mevcut duruma göre bir eğitim adımı (Standart veya Rüya) yürütülür.
3.  **Aktivasyon:** Nöronlar birikmiş girdileri uyarlanabilir aktivasyon fonksiyonu aracılığıyla işler.
4.  **Yayılım:** Nöronlar sıfırlanır ve çıktıları bağlantılara iletir.
5.  **İletim:** Bağlantılar çıktıları ağırlıklarla çarpar ve sonucu *bir sonraki* zaman adımı için tamponlar.

Biriktirme ve iletimin bu şekilde ayrılması, yarış koşullarını (race conditions) önler ve sıralı donanım üzerinde paralel işlemeyi simüle eder.

### 3.2. Eğitim Protokolü: İleri Ateşle, İleri Bağla (FFWF)
RealNet, zamansal farkındalığa sahip yerel bir öğrenme kuralı için geri yayılımı terk eder.
*   **Kavram:** "Birlikte Ateşle, Birlikte Bağla" (uzamsal korelasyon) yerine FFWF, "İleri Ateşle, İleri Bağla" (zamansal nedensellik) üzerine odaklanır. Bir nöronun ateşlenmesinin, bir sonraki zaman adımında başka bir nöronun ateşlenmesini *tahmin ettiği* bağlantıları güçlendirir.
*   **Mekanizma:**
    *   **Pozitif Korelasyon:** Nöron A (t-1) pozitif ateşler ve Nöron B (t) pozitif ateşlerse, $W_{AB}$ ağırlığı artırılır.
    *   **Negatif Korelasyon:** Nöron A (t-1) pozitif ateşler ve Nöron B (t) negatif ateşlerse, $W_{AB}$ azaltılır (engelleyici/inhibitory).
    *   **Çürüme (Decay):** Ateşlemeyen nöronlardan gelen veya onlara giden bağlantılar sıfıra doğru çürütülür, ilgisiz yollar budanır.
*   **Ağırlık Patlaması Kontrolü:** Basit ağırlık çürümesi yerine algoritma, ağırlıkları kaynağın hedefe olan *dolaylı* katkısına göre ayarlayarak kontrolden çıkan geri besleme döngülerini önler.

### 3.3. Rüya Eğitimi (Damıtma/Distillation)
Her adımda açık bir denetim olmadan yakınsamak (converge) için RealNet "Rüya Eğitimi"ni kullanır.
*   **Süreç:** Ağ periyodik olarak dış girdiden koparılır. Çıktı nöronları istenen değerlere (bir veri setinden) sabitlenir/kilitlenir.
*   **Damıtma:** Ağ iç döngüler çalıştırır. FFWF algoritması bu "rüya" durumlarını geriye doğru (nedensel olarak) yayar ve doğal olarak bu çıktılara yol açacak yolları güçlendirir.
*   **Temellendirme (Grounding):** Bu süreç, soyut iç temsilleri somut hedef çıktılara temellendirir ve kaotik kısa süreli hafızayı etkili bir şekilde yapılandırılmış uzun süreli ağırlıklara damıtır.

## 4. Yakınsama ve Kararlılık
RealNet'te yakınsama, zamansal girdi örüntülerini istenen çıktı durumlarına eşleyen kararlı, öngörücü yolların oluşumu olarak tanımlanır.
*   **Öz-Düzenleme:** Uyarlanabilir aktivasyon fonksiyonu, aşırı aktif nöronları doğal olarak sönümler.
*   **Budama:** FFWF algoritması zayıf bağlantıları sürekli olarak budayarak ağı seyreltir (sparsifying).
*   **Gelecek Tahmini:** Ağ, kendi gelecek durumlarını tahmin etmeyi doğal olarak öğrenir ve içsel sürprizi (serbest enerji ilkesi) en aza indirir.

## 5. Vizyon ve Gelecek Yönelimleri

RealNet, "Organik Yapay Zeka"ya doğru bir adımı temsil eder. Sadece statik verileri sınıflandırmak için değil, sürekli bir veri akışında var olmak, deneyimlemek ve uyum sağlamak için tasarlanmıştır.

*   **Ölçeklenebilirlik:** Katmansız doğa, tüm ağı yeniden eğitmeden yeni nöronların sorunsuz bir şekilde eklenmesine izin verir.
*   **Ağlar Arası İletişim:** Birden fazla RealNet doğrudan bağlanabilir, iç durumları ve "düşünceleri" ayrık tokenlara kodlama/kod çözme ihtiyacı olmadan paylaşabilir.
*   **Gerçek Çok Modluluk (Multimodality):** Verileri zaman içindeki ham sinyaller olarak işleyerek RealNet; metin, ses ve videoyu temel olarak aynı şekilde ele alır: öğrenilecek ve tahmin edilecek zamansal örüntüler.

## Lisans

Bu proje MIT Lisansı altında lisanslanmıştır.
