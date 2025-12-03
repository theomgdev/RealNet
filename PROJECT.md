# RealNet

## Açıklama

RealNet, farklı bir Neural Network projesidir. Beyinden ilhamla geliştirilmiş bir mimari sunar. Hebbian learningten ilham alan bir algoritması vardır ama bir o kadar da bilindik mimarilerden farklıdır ve kendine özgü algoritmalarla converge yapar.

## Mimari

Ağ dışardan bakıldığında nöron ve bağlantılardan oluşan full bağlı bir yapıdır. FNN'leri andırır ama mevcut mimarilerden çok farklıdır çünkü bir RealNet'te katmanlar yoktur. Bir nöron diğer bütün nöronlardan bağ alır. Bu kaotik bağlı yapı ağın 2D bir düzlemden 3D, 4D hatta 5D ve fazlasında veri iletimi yapmasına olanak sağlarken, her nöronun diğer bütün nöronlarla bağ kurabilmesi dairesel veri iletim döngülerine olanak sağlar. Bu döngüler verinin nöronların ve bağlantıların oluşturduğu döngüsel yolda sıkışıp kaldığı ve dışarıya periyodik sinyaller verdiği bir tür kısa süreli hafıza gibi çalışır. Uzun süreli hafıza ise inference esnasındaki aktif öğrenme sürecinin(bahsedilecek) bir yan ürünü olarak ortaya çıkar. Bir kaç nöron önceden output ve input olarak işaretlenmiştir. Somut anlamıyla ağda bir yön bulunmasa da soyut ve dolaylı olarak bir yön eğitim sonrası oluşur. Oluşturma esnasında önerilen weight değerleri rastgele olarak -2 ve 2 aralığında önerilir.

### Aktivasyon Fonksiyonları ve Gateleme

Aktivasyon fonksiyonu olarak herhangi bir aktivasyon fonksiyonu denenebilir ama önerilen aktivasyon fonksiyonu algoritma olarak şu şekildedir:
    Her nöron için ortalama ateşleme değeri her ateşleme sonrası +-ateşleme değeri bölü şu ana kadarki ateşleme sayısı(timestep sayısı) olarak ortalamaya eklenerek birikimli olarak ateşlendikçe hesaplanır. Bu ortalama ve ateşleme sayısı sonraki hesaplar için nöron nesnesinde tutulur.
    Her nöron için o ana kadarki maksimum ateşleme değeri de o nöron nesnesinde tutulur.
    Her nöron için o ana kadarki minimum ateşleme değeri de o nöron nesnesinde tutulur.

#### Tek Satırlık Kod İfadesi

(Python, C++, Excel vb. için uyumlu format. `0.000001` sıfıra bölünmeyi önlemek içindir.)

```python
y = tanh( k * (x - x_ort) / ( (x_max - x_min)/2 + (x_max + x_min - 2*x_ort)/2 * (x - x_ort) / (abs(x - x_ort) + 0.000001) ) ) / tanh(k)
```

##### Matematiksel Format

$$y = \frac{\tanh\left( k \cdot \frac{x - x_{ort}}{ \frac{x_{max} - x_{min}}{2} + \frac{x_{max} + x_{min} - 2x_{ort}}{2} \cdot \frac{x - x_{ort}}{|x - x_{ort}| + \epsilon} } \right)}{\tanh(k)}$$

-----

##### Matematiksel Mantığı

Bu formül iki ana mekanizmayı birleştirir:

1.  **Dinamik Ölçeklendirme (Paydadaki karmaşık kısım):**

      * `if-else` kullanmadan $x$'in ortalamanın sağında mı solunda mı olduğunu anlamak için $\frac{x - x_{ort}}{|x - x_{ort}|}$ yapısını kullanırız. Bu ifade $x > x_{ort}$ ise **+1**, $x < x_{ort}$ ise **-1** verir.
      * Bu +1/-1 anahtarı sayesinde, bölen sayı otomatik olarak ya $(x_{max} - x_{ort})$ olur ya da $(x_{ort} - x_{min})$ olur.

2.  **Normalizasyon (En dıştaki `/ tanh(k)`):**

      * Standart $\tanh(k)$ hiçbir zaman tam 1 olmaz (örneğin $\tanh(3) \approx 0.995$).
      * Fonksiyonun sonucunu, hesaplanan bu maksimum değere ($\tanh(k)$) böleriz.
      * Böylece $x$, $x_{max}$ sınırına dayandığında; pay $\tanh(k)$, payda da $\tanh(k)$ olur ve sonuç **kesinlikle 1** çıkar. Bu sayede $k$ çok düşük olsa bile (örn: 1.5) grafik uçlara tam oturur.
      * Önerilen k değeri altın oran yani 3 tür. RealNet'te sabittir.

##### Neden Böyle bir Fonksiyon?

Bu fonksiyon sayesinde nöron zamanla ortalama değerlere az tepki vermeye başlar. Ani spikeları iyi ele alır. Aradaki değerleri de biraz uçlarda max-min'e yakın yorumlayarak daha iyi generalization sağlar. Sürekli aşırı aktive nöronlar ortalamanın max'a yaklaşması ile zamanla az tepki vermeye ve hatta negatif çıktı üretmeye de başlar. Bu da eğitim esnasında bağlarını zayıflatmaları ve yeni bağlar keşfetmeleri ile sonuçlanır. Sürekli aynı değeri veren nöronlar da zamanla 0'a yakın çıktılar vererek etkilerini kaybeder, bağlarını koparır, bu da ağın yeni örüntüler keşfetmesine olanak sağlar. Basitçe her nöron tekrar eden veriye karşı zamanla duyarsızlaşarak kıymetli veriyi filtreleyen bir filtreye dönüşür. Özellikle kısa süreli hafıza sağlayan döngülerin bozulması, yeniden oluşması, eskimiş kullanılmayan ağ ile az etkileşimdeki hafızanın decayi için kritiktir. Neural networkü tekrar eden verilerin ve rastgele döngülerin kaotikliğinden kurtarıp önemli veriye odaklar.

## Algoritmalar

Ağın inference algoritması da training algoritması da kendine özgüdür. RealNet teorik olarak, veriseti olmadan da, veriseti ile de, aktif olarak öğrenebilen, kısa süreli ve uzun süreli hafızası olan, hayal kurabilen bir modeldir. Pratikte çalıştığı bir senaryoda haliyle yapay zeka dünyasının kutsal kasesi olacaktır. Teorik olarak başarıyla converge olur.

### Inference

Ağ standart FNN'ler ve popüler yaklaşımın aksine tek seferde baştan sonra çalıştırılmaz. Her adımda önce her bağlantı önceki timestepten beklettiği sonuç değerini(eğer mevcutsa) hedef nöronun toplamına ekler ve bekletilen değer sıfırlanır. Ardından bir standart training step çalıştırılır(eğer veriseti varsa standart training step yerine dream training step çalıştırılır) ve state hafızada tutulur.(Training bölümünde bahsedilecek) Ardından her nöron kendinden sonraki nöronlara iletilecek veriyi önce aktivasyon fonksiyonundan geçirir ve ardından bağlantılara teslim eder, nöron kendini sıfırlar, bağlantılar bu veriyi weight değeri ile çarpıp sıradaki timestep için bekletir, sıradaki timestep gelmeden veri bağlantıların hedef nöronlarına iletilmez aksi taktirde hedef nöron daha kendini sıfırlamadığından veri karmaşası ve race conditionlar ortaya çıkar. Bütün nöronlar değerlerini bağlantılara teslim edip kendini sıfırladıktan sonra sıradaki timestepte her timestepte olduğu gibi ilk olarak bağlantılarda bekletilen sonuç değerleri hedef nöronlarda toplanır. Adım adım ağda veri böyle ilerler. Veri kimi zaman dairesel döngülere girer, kimi zaman geri döner, kimi zaman ileri gider. Aslınd ağda yön yoktur ama bir kaç nöron önceden output ve input olarak işaretlenmiştir, bu yüzden dolaylı olarak bir yönden bahsedilebilir. Ağın eğitilmediği senaryoda verinin ileri gitmesi ve output nöronlarına ulaşması garanti edilemez. Her bir kaç inference sonrası ağ dream training step ile output olarak işaretlenmiş nöronların kıymetli veriyi distill etmesi için verisetinde grounding yapmalıdır. (dream trainingden bahsedilecek)

### Training

Training algoritması bilindik gradient descent, back-propagation, genetic algorithms veya reinforcement learning algoritmalardan oldukça farklıdır. Ağın eğitim algoritmasına popüler algoritmalar arasında en yakın olanı hebbian learning algoritmasıdır, yine de ağın eğitim algoritması hebbian learningden bile oldukça farklıdır. Hebbian learningden farklı olarak FTWT(fire together wire together) algoritması yerine FFWF(fire forward wire forward) adını koyduğum bir algoritma işler. Buradaki forward kelimesi inference esnasında timestepler arası zaman için kullanılır.

#### Standart Training Step

Her inference timestepinde ağın statei standart training step sonrası bir sonraki timestep için hafızaya alınır. Bir sonraki inference timestepte standard training step esnasında hafızadaki state mevcut state ile kıyaslanır. Önceki timestepte pozitif ateşlenmiş bütün nöronlardan şu anki timestepte pozitif ateşlenmiş bütün nöronlara bağ pozitif yönde biraz güçlendirilir. Önceki timestepte negatif ateşlenmiş nöronlardan şu anki timestepte negatif ateşlenmiş nöronlara da aynı işlem yapılır ve weight pozitif yönde güçlendirilir. Önceki timestepte ateşlenmemiş bütün nöronlardan bu timestepte ateşlenmiş bütün nöronlara ise bağ weight negatiften sıfıra veya pozitiften sıfıra yaklaştırılarak biraz zayıflatılır. Önceki timestepte negatif ateşlenmiş nöronlardan şu anki timestepte pozitif ateşlenmiş nöronlara bağ pozitif weight ise biraz zayıflatılır negatif weight ise negatif yönde güçlendirilir. Önceki timestepte pozitif ateşlenmiş nöronlardan şu anki timestepte negatif ateşlenmiş nöronlara da aynı işlem uygulanır, ve bağ weighti pozitif ise zayıflatılır, bağın weighti sıfır veya negatif ise bağ weighti negatif yönde güçlendirilir. Önceki timestepte ateşlenmemiş nöronlardan şu anki timestepte ateşlenmiş nöronlara bağ weight hangi yöndeyse o yönden sıfıra doğru zayıflatılır. Önceki timestepte ateşlenmiş nöronlardan şu anki timestepte ateşlenmemiş nöronlara da aynı işlem uygulanır ve bağ weight hangi yöndeyse o yönden sıfıra doğru zayıflatılır. Güçlendirme veya zayıflatma miktarı nöronların ateşlenme değerlerinin farkı ile orantılıdır. Fark arttıkça bağ zayıflar, fark azaldıkça bağ güçlenir. Amaç iki timestep arasında negatif veya pozitif korelasyonlu ateşlenmiş nöronları bulup ilerde aynı davranışın pekişmesi için hafifçe bağlamaktır. Haliyle benzer veri geldiğinde benzer durum ortaya çıkar. Zamanla deneyimler birikir ve zaman düzleminde gruplanır. Bu sebep-sonuç ilişkisini güçlendirirken, içsel future prediction sağlar. Düzlem zaman odaklı olduğundan hebbian learningin klasik problemleri bu algoritmada çözülmüştür. Tabi ki tek başına standart training step, verinin doğru şekilde output işaretli nöronlara akmaması yüzünden, converge garanti etmez. Standart training step yalnızca ağın inferencelar arasında dataset olmaksızın aktif gelişimi için ekstra bir algoritmadır. Asıl altyapıyı oluşturan algoritma datasetler ile uygulanan dream training steptir. Özetle:

- Ateşlenmemiş -> Pozitif veya negatif ateşlenmiş = Weight 0'a doğru negatiften veya pozitiften biraz yaklaştırılır
- Pozitif veya negatif ateşlenmiş -> Ateşlenmemiş = Weight 0'a doğru negatiften veya pozitiften biraz yaklaştırılır
- Pozitif ateşlenmiş -> Negatif ateşlenmiş = Weight negatife doğru güçlendirilir, weight pozitifse zayıflatılır.
- Negatif ateşlenmiş -> Pozitif ateşlenmiş = Weight negatife doğru güçlendirilir, weight pozitifse zayıflatılır.
- Pozitif ateşlenmiş -> Pozitif ateşlenmiş = Weight pozitife doğru güçlendirilir, weight negatifse zayıflatılır.
- Negatif ateşlenmiş -> Negatif ateşlenmiş = Weight pozitife doğru güçlendirilir, weight negatifse zayıflatılır.

##### Güçlendirme/Zayıflatma Miktarı

Güçlendirme/zayıflatma miktarı nöronların ateşlenme değerlerinin farkı ile orantılıdır. Fark arttıkça bağ zayıflar, fark azaldıkça bağ güçlenir. Aynı zamanda güçlendirme/zayıflatma miktarı bir öğrenme faktörü ile çarpılarak öğrenme eğrisinin kontrolü sağlanır. Yüksek öğrenme katsayısı ağın inference esnasında yaşadıklarını kalıcı hafızaya (weightlerine) kazıyarak aşırı-öğrenme ile overfit yapmasını ve generalizasyonunu kaybetmesine sebebiyet verebilir. Bu yüzden çok düşük öğrenme katsayısı tavsiye edilir. Verisetindeki kritik noktalarda hafif artan öğrenme katsayısı ve verisetindeki görece önemsiz verilerde hafif azalan öğrenme katsayısı kullanılabilir.

##### Weight ve Value Explosion

Bir bağın sürekli sonsuza kadar negatif veya pozitif yönde güçlenmesi için önceki timestepte kaynak nöronun sürekli korelasyonla zaman ardışık ateşlenmesi gerekir. Aşırı tekrar eden verilerde bu bir problem yaratır. Ayrıca oluşan döngülerde veri tekrar eden benzer ateşlenmelere sebebiyet vererek özellikle döngülerde weight ve value explosiona sebebiyet verir. RealNet'te bu tip durumlar için decay veya weigt normalizasyon kullanılmaz. Bunun yerine standart training step esnasında hedef nöronun mevcut değerinin ne kadarını bir önceki stepteki kaynak kıyaslanan nörondan aldığına bakılır ve bu değerin mutlak değeri hedef nöronun toplam değerinden sıfıra doğru çıkartılarak/toplanarak doğrudan eğitim kaynaklı korelasyon yerine dolaylı korelasyona göre bağların güçlenmesi/zayıflaması sağlanır. Doğrudan bağ kaynaklı korelasyon denklemden çıkınca explosionun meydana gelme olasılığı ciddi derecede düşer.

#### Dream Training Step

Ağın mimarisi rastgele -1 ve 1 arasında weightler ile oluşturulduktan sonra, boş ağ inference ve standart training steplerden önce, converge için dream training denen bir sürece sokulmak zorundadır. Aksi taktirde ağ input ve output olarak işaretlenmiş nöronları bilmediğinden converge olamaz. Dream training sırasında ağın output olarak işaretlenmiş nöronları ağda standart training step esnasında zaman düzleminde gruplanmış ve ayrıksanmış kıymetli veriyi distill etmeyi öğrenir. Her dream training step standart training stepe içerir ve inference esnasında dataset mevcutsa çalıştırılır aksi taktirde inferenceta yalnızca standart training step çalıştırılır. Basitçe dream training step, output nöronlarına datasetten alınan değerler yapay olarak ateşlenmiş gibi koyulduktan sonra çalıştırılan standart training steptir. Bu sayede yapay ateşlenmiş output nöronları önceki timestepte ateşlenmiş nöronlar ile bağ kurarlar, negatif ateşlenmişler ile negatif yönde bağ güçlendirirken pozitif ateşlenmişler ile pozitif yönde bağ güçlendirirler. Önceki timestepte ateşlenmemişler ile bağları 0'a yaklaşarak zayıflarken ateşlenmişler ile güçlenir. Bu sayede ağın output olarak işaretlenmiş nöronları ağ içinde gezen veriden kendisi ile korele olan bilgiyi zamanla güçlenip zayıflayan bağlarla tespit eder ve yazar. Ağın kaç timestepte bir çıktı vermesi isteniyorsa dream training step o sıklıkla çalıştırılır aksi taktirde inputlar verildikten sonra bir kaç timestep boyunca boş inputlar ile yalnızca standart training step çalıştırılır. Dream training step esnasında output olarak işaretlenmiş nöronlar ağ içi ne durumda olursa olsun kıymetli bilgileri bir kaç timestep öncesinden gezen veriden distill edecektir.

### Convergence

Ağ verinin zaman düzleminde gruplanması ve gelişen future-prediction ile oluşan verinin ağın output olarak işaretlenmiş nöronlarına distill edilmesi ile converge olur. Zamanla birikimli zayıflayan ve güçlenen bağlar ağın generalize bir tutum elde etmesini sağlar. RealNet'in özellikle LLM olarak tek token girdi ve tek token çıktı ile çalıştırılması önerilir. Token ihtimalleri 0-1 aralığında uygulanmış bir output clamp ile uca explode olsa bile doğru çıktıyı verir. Bu tip ara değerlerin kritik olmadığı var ya da yok, 0 veya 1 tipi problemler için RealNet daha verimlidir.