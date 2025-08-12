# Customer-Complaints-Text-Analysis

Bu proje, müşteri şikayet metinlerini Python ile sınıflandırmayı ve Power BI ile görselleştirmeyi amaçlamaktadır.  
Veri ön işleme, sınıflandırma modeli ve görselleştirme adımları içerir.

## Kullanılan Teknolojiler
- Python (pandas, scikit-learn, seaborn, matplotlib)
- Pycharm
- Power BI

## Proje Dosya Yapısı
```
├── data/
│   └── complaints.csv        # Şikayet verisi (örnek)
├── notebooks/
│   └── analysis.ipynb        # Analiz ve modelleme Jupyter notebook dosyası
├── scripts/
│   └── complaint_classifier.py  # Python sınıflandırma scripti
├── README.md                 # Proje açıklaması (bu dosya)
└── requirements.txt          # Proje bağımlılıkları
```

## Kurulum ve Çalıştırma
1. Gerekli Python kütüphanelerini aşağıdaki komutlarla yükleyebilirsiniz:
```bash
pip install pandas scikit-learn seaborn matplotlib
pip install scikit-learn
pip install wordcloud
pip install matplotlib seaborn
pip install textblob
pip install plotly
        --NLP için ağaıdaki kurulumlar--
pip install transformers
pip install torch
```

### Kütüphaneler
```
import random
import pandas as pd
```
### Kategoriler
```
categories = [
    "Faturalama",
    "Sistem Hatası",
    "Erişim Sorunu",
    "Yanlış Bilgi",
    "Destek Talebi"
]
```
### Duygular
```
sentiments = ["Pozitif", "Negatif", "Nötr"]
```
### Kategoriye göre örnek şikayet cümleleri
```
complaints_examples = {
    "Faturalama": [
        "Faturam beklediğimden yüksek geldi.",
        "İki kez ödeme alınmış, düzeltilmesini istiyorum.",
        "Fatura detaylarında hatalı kalemler var."
    ],
    "Sistem Hatası": [
        "Uygulama açılırken sürekli hata veriyor.",
        "Sunucu hatası nedeniyle işlemimi tamamlayamadım.",
        "Sistem güncellemesi sonrası giriş yapamıyorum."
    ],
    "Erişim Sorunu": [
        "Hesabıma giriş yapamıyorum.",
        "Web sitesine erişim çok yavaş.",
        "VPN olmadan sisteme bağlanamıyorum."
    ],
    "Yanlış Bilgi": [
        "Destek ekibi yanlış yönlendirme yaptı.",
        "Sözleşmede belirtilmeyen ücret yansıtılmış.",
        "Kampanya bilgileri eksik verilmiş."
    ],
    "Destek Talebi": [
        "Yeni paket hakkında bilgi almak istiyorum.",
        "Kurulum için randevu talep ediyorum.",
        "Servis süresi hakkında detay verir misiniz?"
    ]
}
```
### Veri üretme
```
data = []
num_samples = 500  # Üreteceğimiz veri sayısı

for _ in range(num_samples):
    category = random.choice(categories)
    text = random.choice(complaints_examples[category])
    sentiment = random.choice(sentiments)
    data.append({
        "complaint_text": text,
        "category": category,
        "sentiment": sentiment
    })
```
### DataFrame oluşturma
```
df = pd.DataFrame(data)
```
### CSV'ye kaydetme
```
df.to_csv("complaints_dataset.csv", index=False, encoding="utf-8-sig")

print(f"{num_samples} satırlık sahte şikayet veri seti oluşturuldu: complaints_dataset.csv")
```
### Sınıflandırma Aşaması (Train-Test)
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
```
### Veri setini oku
```
df = pd.read_csv("complaints_dataset.csv")
```

### === 1. KATEGORİ TAHMİNİ ===
```
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
    df["complaint_text"], df["category"], test_size=0.2, random_state=42
)

vectorizer_cat = TfidfVectorizer()
X_train_cat_vec = vectorizer_cat.fit_transform(X_train_cat)
X_test_cat_vec = vectorizer_cat.transform(X_test_cat)

model_cat = LogisticRegression(max_iter=1000)
model_cat.fit(X_train_cat_vec, y_train_cat)

y_pred_cat = model_cat.predict(X_test_cat_vec)

print("=== Kategori Tahmini ===")
print("Doğruluk Oranı:", accuracy_score(y_test_cat, y_pred_cat))
print(classification_report(y_test_cat, y_pred_cat))
```
### === 2. DUYGU TAHMİNİ ===
```
X_train_sent, X_test_sent, y_train_sent, y_test_sent = train_test_split(
    df["complaint_text"], df["sentiment"], test_size=0.2, random_state=42
)

vectorizer_sent = TfidfVectorizer()
X_train_sent_vec = vectorizer_sent.fit_transform(X_train_sent)
X_test_sent_vec = vectorizer_sent.transform(X_test_sent)

model_sent = LogisticRegression(max_iter=1000)
model_sent.fit(X_train_sent_vec, y_train_sent)

y_pred_sent = model_sent.predict(X_test_sent_vec)

print("\n=== Duygu Tahmini ===")
print("Doğruluk Oranı:", accuracy_score(y_test_sent, y_pred_sent))
print(classification_report(y_test_sent, y_pred_sent))
```

### === 3. Model çıktısını CSV olarak kaydetme (Power BI için) ===
```
df["predicted_category"] = model_cat.predict(vectorizer_cat.transform(df["complaint_text"]))
df["predicted_sentiment"] = model_sent.predict(vectorizer_sent.transform(df["complaint_text"]))

df.to_csv("complaints_with_predictions.csv", index=False, encoding="utf-8-sig")
print("\nTahmin sonuçları complaints_with_predictions.csv dosyasına kaydedildi.")
```
### === Görselleştirme ===
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Tahmin sonuçlarını oku

df = pd.read_csv("complaints_with_predictions.csv")

# Tema
sns.set_theme(style="whitegrid")

# === 1. Kategori dağılımı ===

sns.countplot(x="predicted_category", data=df, order=df["predicted_category"].value_counts().index, hue="predicted_category",palette="Set2",legend=False)
plt.title("Şikayet Kategori Dağılımı")
plt.xticks(rotation=30)
plt.show()
plt.savefig("kategori_dagilimi.png")

# === 2. Duygu dağılımı ===
plt.figure(figsize=(6,4))
sns.countplot(x="predicted_sentiment", data=df, order=df["predicted_sentiment"].value_counts().index,hue="predicted_sentiment", palette="coolwarm",legend=False)
plt.title("Duygu Analizi Sonuçları")
plt.show()
plt.savefig("duygu_analizi.png")

# === 3. Kelime bulutu ===
from wordcloud import WordCloud

text = " ".join(df["complaint_text"])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("En Çok Geçen Kelimeler")
plt.show()
plt.savefig("kelime_analiz.png")
```

![kategori_dagilimi.png](kategori_dagilimi.png)
![duygu_analizi.png](duygu_analizi.png)
![kelime_analiz.png](kelime_analiz.png)
```
```

### Plotly ile Etkileşimli Grafik
```
import plotly.express as px

fig = px.histogram(
    df,
    x='predicted_category',
    color='predicted_sentiment',
    barmode='group',    # Gruplandırılmış bar chart
    labels={'predicted_category':'Kategori', 'count':'Şikayet Sayısı', 'predicted_sentiment':'Duygu'},
    title='Kategoriye Göre Duygu Dağılımı'
)
fig.show()
```
### İç içe pasta grafiği
```
import plotly.express as px
fig = px.sunburst(
    df,
    path=['predicted_category', 'predicted_sentiment'],
    title='Kategori ve Duygu İç İçe Görselleştirme'
)
fig.show()
```

