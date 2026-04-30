"""
config.py — BERTopic + XGBoost modülü için merkezi konfigürasyon.

Yollar ve hiperparametreler burada ayarlanır; diğer modüller buradan içe aktarır.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PROJE YOLU HESAPLAMA
# Bu dosya bertopic_xgboost/ içinde; üst dizin araDonemProje/
# ─────────────────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent


@dataclass(frozen=True)
class Paths:
    # Ham preprocess çıktıları (mevcut veriler)
    preprocess_dir: Path = _PROJECT_ROOT / "preprocess"

    # Modül çıktıları
    output_dir: Path = _THIS_DIR / "output"

    # Tüm kategorilerdeki preprocessed CSV'ler (pattern: *_preprocessed.csv)
    preprocessed_csv_pattern: str = "*_preprocessed.csv"

    # Metin kolonu adı (preprocessing.py'nin ürettiği kolon)
    text_column: str = "temiz urun yorum"

    # Puan kolonu
    score_column: str = "puan"


@dataclass(frozen=True)
class BERTopicConfig:
    """BERTopic model ayarları."""

    # Çok dilli, Türkçe için uygun embedding modeli
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # "auto" → BERTopic otomatik, sayı → hedef konu sayısı
    nr_topics: int | str = "auto"

    # Konu başına minimum yorum sayısı
    min_topic_size: int = 20

    # Konu başına gösterilecek kelime sayısı
    top_n_words: int = 10

    # Minimum metin uzunluğu (karakter)
    min_text_len: int = 5


@dataclass(frozen=True)
class XGBoostConfig:
    """XGBoost sınıflandırıcı ayarları."""

    # Eğitim / test bölme oranı
    test_size: float = 0.2

    # Rastgele tohum
    random_state: int = 42

    # XGBoost hiperparametreleri
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    use_label_encoder: bool = False
    eval_metric: str = "mlogloss"

    # Minimum konu örnekleri (daha az örnekli konular XGBoost'tan çıkarılır)
    min_class_samples: int = 10

    # Özellik tipi: "embeddings" | "tfidf" | "both"
    feature_type: str = "embeddings"

    # TF-IDF (feature_type="tfidf" veya "both" ise)
    tfidf_max_features: int = 5000


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    bertopic: BERTopicConfig = field(default_factory=BERTopicConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)

    # 10 kategori — preprocess/ altındaki dosyaların stem adları
    categories: tuple[str, ...] = (
        "airfryer",
        "akilli_saat",
        "erkek_kot",
        "kadin_canta",
        "kadin_tayt",
        "kulaklık",
        "mouse",
        "nevresim",
        "sandalye",
        "spor_ayakkabi",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Varsayılan global config nesnesi (modüller bunu içe aktarır)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = Config()
