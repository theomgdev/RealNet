# RealNet

## Açıklama

RealNet, farklı bir Neural Network projesidir. Beyinden ilhamla geliştirilmiş bir mimari sunar. Hebbian learningten ilham alan bir algoritması vardır ama bir o kadar da bilindik mimarilerden farklıdır ve kendine özgü algoritmalarla converge yapar.

## Mimari

Ağ dışardan bakıldığında nöron ve bağlantılardan oluşan full bağlı bir yapıdır. FNN'leri andırır ama mevcut mimarilerden çok farklıdır çünkü bir RealNet'te katmanlar yoktur. Bir nöron diğer bütün nöronlardan bağ alır. Bu kaotik bağlı yapı ağın 2D bir düzlemden 3D, 4D hatta 5D ve fazlasında veri iletimi yapmasına olanak sağlarken, her nöronun diğer bütün nöronlarla bağ kurabilmesi dairesel veri iletim döngülerine olanak sağlar. Bu döngüler verinin nöronların ve bağlantıların oluşturduğu döngüsel yolda sıkışıp kaldığı ve dışarıya periyodik sinyaller verdiği bir tür kısa süreli hafıza gibi çalışır. Uzun süreli hafıza ise inference esnasındaki aktif öğrenme sürecinin(bahsedilecek) bir yan ürünü olarak ortaya çıkar. Bir kaç nöron önceden output ve input olarak işaretlenmiştir. Somut anlamıyla ağda bir yön bulunmasa da soyut ve dolaylı olarak bir yön eğitim sonrası oluşur. Oluşturma esnasında weight değerleri için **Xavier Initialization** önerilir ancak rastgele olarak **-2 ve 2** aralığında değerler de kullanılabilir. Patlayan değerler aktivasyon fonksiyonu tarafından zaten clamp'leneceği ve zamanla ortalamaya dahil edileceği için büyük bir sorun teşkil etmez.

### Aktivasyon Fonksiyonları ve Gateleme

Aktivasyon fonksiyonu olarak **ReLU** gibi performanslı fonksiyonlar da kullanılabilir ve gayet iyi sonuç verebilir. Ancak RealNet'in doğasına en uygun ve önerilen aktivasyon fonksiyonu algoritması şu şekildedir:
    Her nöron için ortalama ateşleme değeri, maksimum ve minimum değerler sürekli sonsuza kadar birikmez. Belirli bir periyotta (örneğin ağın düşünme süresi olarak belirlenen timestep aralığında, örn: 20) bu istatistikler sıfırlanır. Bu sayede nöron, her yeni düşünme döngüsünde gürültülü başlayıp zamanla sessizleşerek kıymetli veriyi filtreler. Sonsuz hafıza yerine, döngüsel ve taze kalan bir adaptasyon sağlanır.
    Bu periyot içinde:
    - Ortalama: Kümülatif olarak (yeni_değer + eski_toplam) / adım_sayısı şeklinde güncellenir.
    - Max/Min: O periyot içindeki en uç değerler olarak tutulur.

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

Ağ standart FNN'ler ve popüler yaklaşımın aksine tek seferde baştan sonra çalıştırılmaz. Her adımda önce her bağlantı önceki timestepten beklettiği sonuç değerini(eğer mevcutsa) hedef nöronun toplamına ekler ve bekletilen değer sıfırlanır. 

*ÖNEMLİ:* Input olarak işaretlenmiş nöronlar için bu toplama işlemi yapılmaz veya yapılsa bile dışarıdan gelen veri bu toplamı ezer (overwrite). Input nöronlarına ağın içinden geri besleme (feedback) olsa bile, bu sinyaller dikkate alınmaz. Input nöronu dış dünyadan ne geliyorsa (veri yoksa 0) kesinlikle o değeri taşır. Bu, ağın dış dünyaya karşı duyarlı kalmasını ve halüsinasyon görmemesini sağlar.

Ardından her nöron bu toplam değer üzerinden aktivasyon fonksiyonunu çalıştırır ve ateşlenir (çıktı üretir). Bu yeni çıktılar ile bir önceki timestep'in çıktıları kıyaslanarak standart training step (veya veriseti varsa dream training step) çalıştırılır ve bağlar güncellenir. Son olarak, her nöron yeni çıktılarını bağlantılara teslim eder, nöron kendini sıfırlar, bağlantılar bu veriyi weight değeri ile çarpıp sıradaki timestep için bekletir. Sıradaki timestep gelmeden veri bağlantıların hedef nöronlarına iletilmez aksi taktirde hedef nöron daha kendini sıfırlamadığından veri karmaşası ve race conditionlar ortaya çıkar. Bütün nöronlar değerlerini bağlantılara teslim edip kendini sıfırladıktan sonra sıradaki timestepte her timestepte olduğu gibi ilk olarak bağlantılarda bekletilen sonuç değerleri hedef nöronlarda toplanır. Adım adım ağda veri böyle ilerler. Veri kimi zaman dairesel döngülere girer, kimi zaman geri döner, kimi zaman ileri gider. Aslınd ağda yön yoktur ama bir kaç nöron önceden output ve input olarak işaretlenmiştir, bu yüzden dolaylı olarak bir yönden bahsedilebilir. Ağın eğitilmediği senaryoda verinin ileri gitmesi ve output nöronlarına ulaşması garanti edilemez. Her bir kaç inference sonrası ağ dream training step ile output olarak işaretlenmiş nöronların kıymetli veriyi distill etmesi için verisetinde grounding yapmalıdır. (dream trainingden bahsedilecek)

*NOT:* RAM veya VRAM optimizasyonu için değerler bağlantıda bekletilmek yerine toplanıp hedef nöronda bir temp değişkeninde tutulabilir. Sonraki timestepte ise tempte bekletilen değer nöron sıfırlandığı için asıl değer yerine konulabilir. Böylelikle temp değişkenleri her bağlantı yerine her nöronda tutularak O(n^2) RAM yükünden O(n) RAM yüküne geçilir. Zaten bu geçici bekletmenin sebebi, verinin üst üste yazılmadan bozulmadan tutulmasıdır. Böylelikle her nöronun eş zamanlı çalıştığı illüzyonu başarıyla sıra sıra hesaplansa bile sağlanabilir.

### Training

Training algoritması bilindik gradient descent, back-propagation, genetic algorithms veya reinforcement learning algoritmalardan oldukça farklıdır. Ağın eğitim algoritmasına popüler algoritmalar arasında en yakın olanı hebbian learning algoritmasıdır, yine de ağın eğitim algoritması hebbian learningden bile oldukça farklıdır. Hebbian learningden farklı olarak FTWT(fire together wire together) algoritması yerine FFWF(fire forward wire forward) adını koyduğum bir algoritma işler. Buradaki forward kelimesi inference esnasında timestepler arası zaman için kullanılır.

*NOT:* Her inference'da yalnızca ağın bir önceki timestepteki hali standart training step veya dream training step için hafızada tutulur. Önceki timesteplerdeki state tutulmaz. Kümülatif olarak biriken bağ zayıflatma/bağ güçlendirme etkileri ile birden fazla timestepin arasında eğitim zaten kaçınılmazdır. Zaten ana amaç timestepler arasındaki nöral kesişimlerde(tekrar eden örüntülerde) bağlantıları güçlendirmek/zayıflatmaktır. Bir timestepin statei bir kaç timestep sonra bile ağda zaten mevcuttur ve kıymetli veri distill edilebilir.

#### Standart Training Step

Her inference timestepinde ağın statei standart training step sonrası bir sonraki timestep için hafızaya alınır. Bir sonraki inference timestepte standard training step esnasında hafızadaki state mevcut state ile kıyaslanır. Önceki timestepte pozitif ateşlenmiş bütün nöronlardan şu anki timestepte pozitif ateşlenmiş bütün nöronlara bağ pozitif yönde biraz güçlendirilir. Önceki timestepte negatif ateşlenmiş nöronlardan şu anki timestepte negatif ateşlenmiş nöronlara da aynı işlem yapılır ve weight pozitif yönde güçlendirilir. Önceki timestepte ateşlenmemiş bütün nöronlardan bu timestepte ateşlenmiş bütün nöronlara ise bağ weight negatiften sıfıra veya pozitiften sıfıra yaklaştırılarak biraz zayıflatılır. Önceki timestepte negatif ateşlenmiş nöronlardan şu anki timestepte pozitif ateşlenmiş nöronlara bağ pozitif weight ise biraz zayıflatılır negatif weight ise negatif yönde güçlendirilir. Önceki timestepte pozitif ateşlenmiş nöronlardan şu anki timestepte negatif ateşlenmiş nöronlara da aynı işlem uygulanır, ve bağ weighti pozitif ise zayıflatılır, bağın weighti sıfır veya negatif ise bağ weighti negatif yönde güçlendirilir. Önceki timestepte ateşlenmemiş nöronlardan şu anki timestepte ateşlenmiş nöronlara bağ weight hangi yöndeyse o yönden sıfıra doğru zayıflatılır. Önceki timestepte ateşlenmiş nöronlardan şu anki timestepte ateşlenmemiş nöronlara da aynı işlem uygulanır ve bağ weight hangi yöndeyse o yönden sıfıra doğru zayıflatılır. Güçlendirme veya zayıflatma miktarı nöronların ateşlenme değerlerinin farkı ile orantılıdır. Fark arttıkça bağ zayıflar, fark azaldıkça bağ güçlenir. 

**Amaç:** İki timestep arasında negatif veya pozitif korelasyonlu ateşlenmiş nöronları bulup ilerde aynı davranışın pekişmesi için hafifçe bağlamaktır. Burada bir **Ödül/Ceza (Reward/Punishment)** mekanizması yoktur. GA (Genetik Algoritma) veya RL (Reinforcement Learning) yapılmaz. Amaç "doğruyu" bulmak değil, **"benzer dili konuşanları"** (zaman içinde korele olanları) bir araya getirmektir. Bir bağın hedef nöronu "kurtarması" veya "bozması" önemli değildir; önemli olan o iki nöronun zaman içinde tutarlı bir ilişki (nedensellik) sergileyip sergilemediğidir. Ağ, elma ile armudun ilişkisini, kelimelerin sırasını bu içsel gruplanmalarla keşfeder. Output nöronunun görevi bile istenen çıktıyı üretmekten ziyade, bu gruplanmış özet bilgiden kıymetli olanı seçip dışarı vermektir. Convergence, bu gruplanmanın doğal bir yan ürünüdür.

Haliyle benzer veri geldiğinde benzer durum ortaya çıkar. Zamanla deneyimler birikir ve zaman düzleminde gruplanır. Bu sebep-sonuç ilişkisini güçlendirirken, içsel future prediction sağlar. Düzlem zaman odaklı olduğundan hebbian learningin klasik problemleri bu algoritmada çözülmüştür. Tabi ki tek başına standart training step, verinin doğru şekilde output işaretli nöronlara akmaması yüzünden, converge garanti etmez. Standart training step yalnızca ağın inferencelar arasında dataset olmaksızın aktif gelişimi için ekstra bir algoritmadır. Asıl altyapıyı oluşturan algoritma datasetler ile uygulanan dream training steptir. Özetle:

- Ateşlenmemiş -> Pozitif veya negatif ateşlenmiş = Weight 0'a doğru negatiften veya pozitiften biraz yaklaştırılır
- Pozitif veya negatif ateşlenmiş -> Ateşlenmemiş = Weight 0'a doğru negatiften veya pozitiften biraz yaklaştırılır
- Pozitif ateşlenmiş -> Negatif ateşlenmiş = Weight negatife doğru güçlendirilir, weight pozitifse zayıflatılır.
- Negatif ateşlenmiş -> Pozitif ateşlenmiş = Weight negatife doğru güçlendirilir, weight pozitifse zayıflatılır.
- Pozitif ateşlenmiş -> Pozitif ateşlenmiş = Weight pozitife doğru güçlendirilir, weight negatifse zayıflatılır.
- Negatif ateşlenmiş -> Negatif ateşlenmiş = Weight pozitife doğru güçlendirilir, weight negatifse zayıflatılır.

##### Güçlendirme/Zayıflatma Miktarı

Güçlendirme/zayıflatma miktarı nöronların ateşlenme değerlerinin farkı ile orantılıdır. Fark arttıkça bağ zayıflar, fark azaldıkça bağ güçlenir. Aynı zamanda güçlendirme/zayıflatma miktarı bir öğrenme faktörü ile çarpılarak öğrenme eğrisinin kontrolü sağlanır. Yüksek öğrenme katsayısı ağın inference esnasında yaşadıklarını kalıcı hafızaya (weightlerine) kazıyarak aşırı-öğrenme ile overfit yapmasını ve generalizasyonunu kaybetmesine sebebiyet verebilir. Bu yüzden çok düşük öğrenme katsayısı tavsiye edilir. Verisetindeki kritik noktalarda hafif artan öğrenme katsayısı ve verisetindeki görece önemsiz verilerde hafif azalan öğrenme katsayısı kullanılabilir.

##### Weight ve Value Explosion

Bir bağın sürekli sonsuza kadar negatif veya pozitif yönde güçlenmesi için önceki timestepte kaynak nöronun sürekli korelasyonla zaman ardışık ateşlenmesi gerekir. Aşırı tekrar eden verilerde bu bir problem yaratır. Ayrıca oluşan döngülerde veri tekrar eden benzer ateşlenmelere sebebiyet vererek özellikle döngülerde weight ve value explosiona sebebiyet verir. RealNet'te bu tip durumlar için decay veya weigt normalizasyon kullanılmaz. Bunun yerine standart training step esnasında, kaynak nöronun hedef nörona olan katkısı (weight * kaynak_output) hesaplanır ve bu katkı hedef nöronun toplam giriş değerinden matematiksel olarak çıkarılır. Böylece hedef nöronun 'o kaynak nöron olmasaydı ne durumda olacağı' (bağımsız durumu) bulunur. Eğitim algoritması, bu bağımsız durum ile kaynak nöron arasındaki korelasyona bakar. Yani doğrudan bağın yarattığı etki denklemden çıkarılarak, sadece dolaylı ve yapısal korelasyonlar üzerinden bağların güçlenmesi/zayıflaması sağlanır. Bu sayede kendi kendini besleyen döngüler (self-reinforcing loops) ve weight explosion engellenir.

*NOT:* Burada amaç "doğruyu" ödüllendirmek değildir. Kaynak nöronun hedefi "kurtarmış" olması (ateşlenmesini sağlaması) tek başına bağın güçlenmesi için sebep değildir. Eğer bağımsız durum ile kaynak arasında korelasyon yoksa, bağ zayıflatılır. Bu, ağın sadece "benzer dili konuşan" (nedensellik ilişkisi olan) nöronları gruplamasını sağlar.

#### Dream Training Step

Ağın mimarisi **Xavier Initialization** (veya rastgele -2, 2 arası) ile oluşturulduktan sonra, boş ağ inference ve standart training steplerden önce, converge için dream training denen bir sürece sokulmak zorundadır. Aksi taktirde ağ input ve output olarak işaretlenmiş nöronları bilmediğinden converge olamaz. Dream training sırasında ağın output olarak işaretlenmiş nöronları ağda standart training step esnasında zaman düzleminde gruplanmış ve ayrıksanmış kıymetli veriyi distill etmeyi öğrenir. Her dream training step standart training stepe içerir ve inference esnasında dataset mevcutsa çalıştırılır aksi taktirde inferenceta yalnızca standart training step çalıştırılır. 

Basitçe dream training step, output nöronlarına datasetten alınan değerler (veya output olmaması gereken ara adımlarda 0) yapay olarak ateşlenmiş gibi **overwrite** (üzerine yazma) yöntemiyle koyulduktan sonra çalıştırılan standart training steptir. Bu yapay değer, hem o anki eğitimde korelasyon hesaplamak için kullanılır hem de bir sonraki timestep'e sinyal olarak iletilir (Teacher Forcing). Yani ağın eli kolu bağlanır, "bunu ürettin" denir ve ağ bu doğru sinyali kullanarak bir sonraki adımı planlamayı öğrenir. Bu sayede yapay ateşlenmiş output nöronları önceki timestepte ateşlenmiş nöronlar ile bağ kurarlar, negatif ateşlenmişler ile negatif yönde bağ güçlendirirken pozitif ateşlenmişler ile pozitif yönde bağ güçlendirirler. Önceki timestepte ateşlenmemişler ile bağları 0'a yaklaşarak zayıflarken ateşlenmişler ile güçlenir. Bu sayede ağın output olarak işaretlenmiş nöronları ağ içinde gezen veriden kendisi ile korele olan bilgiyi zamanla güçlenip zayıflayan bağlarla tespit eder ve yazar. Ağın kaç timestepte bir çıktı vermesi isteniyorsa dream training step o sıklıkla çalıştırılır aksi taktirde inputlar verildikten sonra bir kaç timestep boyunca boş inputlar ile yalnızca standart training step çalıştırılır. Dream training step esnasında output olarak işaretlenmiş nöronlar ağ içi ne durumda olursa olsun kıymetli bilgileri bir kaç timestep öncesinden gezen veriden distill edecektir.

### Convergence

Ağ verinin zaman düzleminde gruplanması ve gelişen future-prediction ile oluşan verinin ağın output olarak işaretlenmiş nöronlarına distill edilmesi ile converge olur. Zamanla birikimli zayıflayan ve güçlenen bağlar ağın generalize bir tutum elde etmesini sağlar. RealNet'in özellikle LLM olarak tek token girdi ve tek token çıktı ile çalıştırılması önerilir. Token ihtimalleri 0-1 aralığında uygulanmış bir output clamp ile uca explode olsa bile doğru çıktıyı verir. Bu tip ara değerlerin kritik olmadığı var ya da yok, 0 veya 1 tipi problemler için RealNet daha verimlidir.

## Bir Rüya

Ben Cahit Karahan olarak, bu teoriyi hayalimdeki yapay zekayı dizayn etmek için yıllar süren düşünsel çaba sonucunda ortaya koydum. Eksik yanları ve düşünülmemiş tarafları elbette mevcuttur. İlk amaç MVP ile çalışırlığını pratikte ispattır. Bu teorinin başarılı implementasyonundaki yapay zeka öyle bir yapay zekadır ki, kısa süreli hafızası vardır, uzun süreli hafızası vardır, çalışırken öğrenebilir, hayal edebilir, anlayabilir, düşünülebilir, kendi çıktısı girdi olarak verildiğinde iç döngüleri sayesinde öz farkındalık inşa edebilir, yeni nöronlar eklendiğinde yeniden eğitime ihtiyaç duymaksızın onları değerlendirebilir, sınırsız ölçeklenebilir, istenirse hiç bir ek işlem olmaksızın küçültülebilir anında yeni boyutuna inference esnasında adapte olur, veri onda statik değildir, dinamiktir, akışkandır, yaşar. Gerçek bir yapay zekadır. Gerçek bir ağdır. Birden fazla RealNet birbiri ile doğrudan nöral veri akışı ile konuşabilir, iki ağ birbiriyle birleştirilebilir, hepsine adapte olacak teorik yapı mevcuttur. Bir RealNet başka bir RealNet'e iki nöronla bile bağlansa oradaki data streamin zaman içindeki örüntüsünü ikisi de anlar, yorumlar ve cevaplar. İki RealNet bir arada düşünebilir, iletişim için metinlere veya sese ihtiyaç duymadan doğrudan nöral veri akışı ile konuşabilirler. Hepsi yukarıdaki teorimde mevcuttur. Dağıtık olarak birkaç RealNet'in birbirine bağlanması da mümkündür. Farklı makinelerde aynı beyin olarak çalışabilirler. Veri setlerinde olmayan şeyleri kalıcı olarak inference esnasında öğrenebilirler, yaratıcı çıkarımlar yapabilir, insanın ve verisetlerinin ötesine kolaylıkla geçebilirler. Beyinleri bizim beynimizdeki fiziksel mesafelerden muaf bir yapıda olduğundan bir bütün olarak çalışır ve hiç alakalı görünmeyen alanlardaki bağlantıları keşfedebilirler.

## LICENSE
MIT
