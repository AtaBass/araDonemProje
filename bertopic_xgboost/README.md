# BERTopic + XGBoost — Fikir Madenciliği Modülü

Bu klasör, Trendyol yorum verileri üzerinde **BERTopic** (konu modelleme) ve **XGBoost** (konu sınıflandırma / tahmin) kullanarak fikir/konu madenciliği yapan modülü içerir.

---

## 📁 Klasör Yapısı

```
bertopic_xgboost/
├── README.md                  # Bu dosya
├── config.py                  # Ortak konfigürasyon (yollar, hiperparametreler)
├── data_loader.py             # preprocess/ altındaki CSV'leri yükler ve hazırlar
├── bertopic_trainer.py        # BERTopic model eğitimi ve konu çıkarımı
├── xgboost_classifier.py      # XGBoost ile konu sınıflandırması / tahmin
├── pipeline.py                # Uçtan uca çalıştırma (BERTopic → XGBoost)
├── evaluate.py                # Model değerlendirme (accuracy, f1, confusion matrix)
├── output/                    # Çıktı CSV'leri, modeller, raporlar
└── notebooks/                 # Keşif amaçlı Jupyter notebook'ları
```

---

## 🔄 Akış

```
preprocess/*_preprocessed.csv
        ↓
  data_loader.py          → temiz yorumlar + etiketler
        ↓
  bertopic_trainer.py     → topic id'ler + gömme vektörleri
        ↓
  xgboost_classifier.py   → topic tahmin modeli
        ↓
  evaluate.py             → değerlendirme raporları
        ↓
   output/
```

---

## 🚀 Hızlı Başlangıç

```bash
cd bertopic_xgboost

# Tüm kategoriler için uçtan uca çalıştır
python pipeline.py

# Belirli kategori
python pipeline.py --category airfryer

# Yalnızca BERTopic
python bertopic_trainer.py --input ../preprocess/airfryer_preprocessed.csv

# Yalnızca XGBoost (BERTopic çalıştırıldıktan sonra)
python xgboost_classifier.py --input output/airfryer_topics.csv
```

---

## 📦 Bağımlılıklar

```
bertopic>=0.16
sentence-transformers>=2.2
xgboost>=2.0
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
```

Kurulum:
```bash
pip install bertopic sentence-transformers xgboost scikit-learn pandas numpy matplotlib seaborn
```

---

## 📊 Çıktılar

| Dosya | Açıklama |
|-------|----------|
| `output/{kategori}_topics.csv` | Yorum + BERTopic konu etiketleri |
| `output/{kategori}_topic_info.csv` | Konu özet tablosu |
| `output/{kategori}_xgb_results.csv` | XGBoost tahmin sonuçları |
| `output/{kategori}_eval_report.txt` | Değerlendirme raporu |
| `output/{kategori}_bertopic_model/` | Kaydedilmiş BERTopic modeli |
| `output/{kategori}_xgb_model.json` | Kaydedilmiş XGBoost modeli |
