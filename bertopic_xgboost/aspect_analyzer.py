"""
aspect_analyzer.py — Dinamik Aspect Çıkarımı + Puanlama + Özet Üretimi

Özellikler:
  • BERTopic gerektirmeyen TF-IDF + kümeleme tabanlı dinamik aspect discovery
  • Her aspect için sentiment skor (puan tabanlı)
  • Türkçe doğal dil özet üretimi
  • JSON çıktısı (web sitesi için)

Kullanım:
    python aspect_analyzer.py --category erkek_kot
    python aspect_analyzer.py --all
    python aspect_analyzer.py --category airfryer --output-json web/data/airfryer.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# YOLLAR
# ─────────────────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
PREPROCESS_DIR = _PROJECT_ROOT / "preprocess"
WEB_DATA_DIR = _THIS_DIR / "web" / "data"

CATEGORIES = [
    "airfryer", "akilli_saat", "erkek_kot", "kadin_canta",
    "kadin_tayt", "kulaklık", "mouse", "nevresim", "sandalye", "spor_ayakkabi",
]

CATEGORY_LABELS = {
    "airfryer": "Airfryer",
    "akilli_saat": "Akıllı Saat",
    "erkek_kot": "Erkek Kot Pantolon",
    "kadin_canta": "Kadın Çanta",
    "kadin_tayt": "Kadın Tayt",
    "kulaklık": "Kablosuz Kulaklık",
    "mouse": "Mouse",
    "nevresim": "Nevresim Takımı",
    "sandalye": "Sandalye",
    "spor_ayakkabi": "Spor Ayakkabı",
}

# ─────────────────────────────────────────────────────────────────────────────
# TÜRKÇELEŞTİRME
# ─────────────────────────────────────────────────────────────────────────────
_TR_MAP = str.maketrans("İIĞÜŞÖÇ", "iığüşöç")

def tr_lower(s: str) -> str:
    return str(s).translate(_TR_MAP).lower()

_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002702-\U000027B0\U0000200D\U0000FE0F]+",
    flags=re.UNICODE,
)

def clean_text(s: str) -> str:
    s = _EMOJI_RE.sub(" ", str(s))
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\d+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

# ─────────────────────────────────────────────────────────────────────────────
# TÜRKÇE STOP WORDS
# ─────────────────────────────────────────────────────────────────────────────
STOP_WORDS = {
    "bir", "bu", "da", "de", "için", "ile", "ve", "ama", "çok", "en",
    "ben", "sen", "o", "biz", "siz", "onlar", "mi", "mu", "mı", "mü",
    "var", "yok", "ki", "ne", "olan", "oldu", "olur", "gibi", "daha",
    "hem", "ya", "ise", "bile", "kadar", "göre", "çünkü", "eğer",
    "nasıl", "neden", "sadece", "herkes", "hiç", "artık", "bana",
    "sana", "bize", "size", "onlara", "benden", "senden"
    "ürün", "urun", "aldım", "aldik", "aldim", "alım", "alindi",
    "geldi", "geldik", "beğendim", "begendim", "begendik", "beğendik",
    "çok", "teşekkür", "tesekkur", "memnun", "iyi", "güzel", "guzel",
    "oldu", "tam", "beden", "alın", "alabilirsiniz", "tavsiye", "ederim",
    "herkese", "kesinlikle", "gerçekten", "gercekten", "sipariş",
}

# ─────────────────────────────────────────────────────────────────────────────
# ASPECT SEED KELİMELERİ (dinamik keşif için başlangıç ipuçları)
# Bu liste sadece aspect isimlerini belirlemek için kullanılır;
# gerçek aspect grupları TF-IDF + kümeleme ile otomatik bulunur.
# ─────────────────────────────────────────────────────────────────────────────
ASPECT_SEEDS = {
    "Kumaş & Malzeme": ["kumaş", "kumasi", "kuması", "malzeme", "dokus", "dokusu", "likralı", "likrali", "esnek", "yumuşak", "yumusak", "kalın", "kalin", "ince"],
    "Kalite & Dayanıklılık": ["kalite", "kaliteli", "sağlam", "sagalm", "dayanıklı", "dayanikli", "işçilik", "iscilik", "mükemmel", "mukemmel"],
    "Beden & Kalıp": ["beden", "kalıp", "kalip", "ölçü", "olcu", "numara", "dar", "bol", "geniş", "genis", "tam", "uydu", "oturdu"],
    "Fiyat & Değer": ["fiyat", "fiyatı", "ücret", "ucuz", "pahalı", "pahali", "uygun", "ekonomik", "değer", "deger", "performans"],
    "Kargo & Teslimat": ["kargo", "teslimat", "teslim", "hızlı", "hizli", "paketleme", "paket", "ulaştı", "ulasti", "geldi"],
    "Renk & Görünüm": ["renk", "rengi", "renkli", "görünüm", "gorunum", "görsel", "gorsel", "şık", "sik", "estetik", "tasarım", "tasarim"],
    "Rahat Kullanım": ["rahat", "konfor", "konforlu", "hafif", "giyim", "giyiniyor", "kullanım", "kullanim"],
    "Satıcı & Hizmet": ["satıcı", "satici", "hizmet", "mağaza", "magaza", "firma", "hediye"],
}

# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT: puana göre
# ─────────────────────────────────────────────────────────────────────────────
def score_to_sentiment(score: float) -> str:
    if score >= 4.5:
        return "positive"
    elif score >= 3.0:
        return "neutral"
    else:
        return "negative"

def score_to_label(score: float) -> str:
    if score >= 4.5:
        return "Olumlu"
    elif score >= 3.5:
        return "Genel Olumlu"
    elif score >= 2.5:
        return "Karışık"
    else:
        return "Olumsuz"

# ─────────────────────────────────────────────────────────────────────────────
# VERİ YÜKLEME
# ─────────────────────────────────────────────────────────────────────────────
def load_csv(category: str) -> pd.DataFrame:
    path = PREPROCESS_DIR / f"{category}_preprocessed.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df["temiz urun yorum"] = df["temiz urun yorum"].fillna("").str.strip()
    df["puan"] = pd.to_numeric(df.get("puan", pd.Series(dtype=float)), errors="coerce").fillna(3.0)
    df = df[df["temiz urun yorum"].str.len() >= 10].copy()
    df.reset_index(drop=True, inplace=True)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# DİNAMİK ASPECT ÇIKARIMI
# ─────────────────────────────────────────────────────────────────────────────
def extract_dynamic_aspects(df: pd.DataFrame, n_aspects: int = 8) -> dict[str, list[int]]:
    """
    TF-IDF + KMeans ile dinamik aspect keşfi.
    Returns: {aspect_name: [row_indices]}
    """
    texts = [tr_lower(clean_text(t)) for t in df["temiz urun yorum"].tolist()]

    # TF-IDF vektörleştirme
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.85,
        token_pattern=r"(?u)\b[a-züğışöçıİĞÜŞÖÇ]{3,}\b",
        sublinear_tf=True,
    )
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        return {}

    feature_names = vectorizer.get_feature_names_out()

    # KMeans kümeleme
    n_clusters = min(n_aspects, len(texts) // 5, 12)
    n_clusters = max(n_clusters, 3)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=100)
    labels = km.fit_predict(X)

    # Her küme için en ayırt edici kelimeleri bul
    cluster_terms: dict[int, list[str]] = {}
    for cluster_id in range(n_clusters):
        center = km.cluster_centers_[cluster_id]
        top_idx = np.argsort(center)[::-1][:15]
        top_terms = [feature_names[i] for i in top_idx]
        # stop word filtrele
        top_terms = [t for t in top_terms if t not in STOP_WORDS and len(t) > 3]
        cluster_terms[cluster_id] = top_terms

    # Kümelere aspect ismi ata (seed eşleşmesi ile)
    cluster_aspect: dict[int, str] = {}
    used_aspects: set[str] = set()

    for cluster_id, terms in cluster_terms.items():
        best_aspect = None
        best_score = 0
        for aspect_name, seeds in ASPECT_SEEDS.items():
            score = sum(1 for t in terms if any(s in t or t in s for s in seeds))
            if score > best_score:
                best_score = score
                best_aspect = aspect_name

        if best_aspect and best_aspect not in used_aspects and best_score >= 1:
            cluster_aspect[cluster_id] = best_aspect
            used_aspects.add(best_aspect)
        else:
            # En sık geçen terimi aspect ismi olarak kullan
            if terms:
                label = terms[0].capitalize()
                cluster_aspect[cluster_id] = label
            else:
                cluster_aspect[cluster_id] = f"Konu {cluster_id + 1}"

    # Aspect → satır endeksleri
    aspect_indices: dict[str, list[int]] = defaultdict(list)
    for idx, cluster_id in enumerate(labels):
        aspect_name = cluster_aspect.get(cluster_id, f"Konu {cluster_id}")
        aspect_indices[aspect_name].append(idx)

    # En az 10 yorumu olan aspectleri tut
    aspect_indices = {k: v for k, v in aspect_indices.items() if len(v) >= 10}

    # Her aspect için üst kelimeleri de döndür (görüntüleme için)
    result_with_terms: dict[str, Any] = {}
    for cluster_id, aspect_name in cluster_aspect.items():
        if aspect_name in aspect_indices:
            result_with_terms[aspect_name] = {
                "indices": aspect_indices[aspect_name],
                "top_terms": cluster_terms[cluster_id][:8],
            }

    return result_with_terms


# ─────────────────────────────────────────────────────────────────────────────
# ASPECT PUANLAMA
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AspectResult:
    name: str
    review_count: int
    avg_score: float
    sentiment: str
    sentiment_label: str
    top_terms: list[str]
    positive_pct: float
    negative_pct: float
    neutral_pct: float
    sample_reviews: list[str]
    icon: str = "💬"


ASPECT_ICONS = {
    "Kumaş": "🧵", "Malzeme": "🧵", "Kumaş & Malzeme": "🧵",
    "Kalite": "⭐", "Dayanıklılık": "🔩", "Kalite & Dayanıklılık": "⭐",
    "Beden": "📏", "Kalıp": "📏", "Beden & Kalıp": "📏",
    "Fiyat": "💰", "Değer": "💰", "Fiyat & Değer": "💰",
    "Kargo": "📦", "Teslimat": "📦", "Kargo & Teslimat": "📦",
    "Renk": "🎨", "Görünüm": "🎨", "Renk & Görünüm": "🎨",
    "Rahat": "😌", "Kullanım": "😌", "Rahat Kullanım": "😌",
    "Satıcı": "🏪", "Hizmet": "🏪", "Satıcı & Hizmet": "🏪",
    "Pişirme": "🍳", "Ses": "🔊", "Batarya": "🔋",
    "Bağlantı": "📡", "Ekran": "📱", "Ergonomi": "🖱️",
}

def get_icon(aspect_name: str) -> str:
    for key, icon in ASPECT_ICONS.items():
        if key.lower() in aspect_name.lower():
            return icon
    return "💬"


def compute_aspect_result(
    df: pd.DataFrame,
    aspect_name: str,
    indices: list[int],
    top_terms: list[str],
) -> AspectResult:
    subset = df.iloc[indices]
    scores = subset["puan"].values

    avg_score = float(np.mean(scores))
    pos = float(np.mean(scores >= 4.5))
    neg = float(np.mean(scores <= 2.0))
    neu = 1.0 - pos - neg

    # Örnek yorumlar (en uzun 3 tanesini al)
    sample = (
        subset["temiz urun yorum"]
        .sort_values(key=lambda x: x.str.len(), ascending=False)
        .head(3)
        .tolist()
    )

    return AspectResult(
        name=aspect_name,
        review_count=len(indices),
        avg_score=round(avg_score, 2),
        sentiment=score_to_sentiment(avg_score),
        sentiment_label=score_to_label(avg_score),
        top_terms=top_terms,
        positive_pct=round(pos * 100, 1),
        negative_pct=round(neg * 100, 1),
        neutral_pct=round(neu * 100, 1),
        sample_reviews=sample,
        icon=get_icon(aspect_name),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ÜRÜN ÖZETİ ÜRETİMİ (kural tabanlı Türkçe)
# ─────────────────────────────────────────────────────────────────────────────
def generate_summary(
    category_label: str,
    aspects: list[AspectResult],
    overall_score: float,
    total_reviews: int,
) -> str:
    positive = [a for a in aspects if a.avg_score >= 4.2]
    negative = [a for a in aspects if a.avg_score < 3.2]
    mixed = [a for a in aspects if 3.2 <= a.avg_score < 4.2]

    lines: list[str] = []

    # Genel giriş
    if overall_score >= 4.5:
        lines.append(
            f"{category_label} kategorisi, {total_reviews} müşteri yorumu analiz edilerek "
            f"değerlendirilmiştir. Genel memnuniyet düzeyi oldukça yüksektir."
        )
    elif overall_score >= 4.0:
        lines.append(
            f"{category_label} kategorisi, {total_reviews} müşteri yorumu analiz edilerek "
            f"değerlendirilmiştir. Ürünler genel itibarıyla olumlu karşılanmaktadır."
        )
    elif overall_score >= 3.0:
        lines.append(
            f"{category_label} kategorisi, {total_reviews} müşteri yorumu analiz edilerek "
            f"değerlendirilmiştir. Müşteri görüşleri karışık bir tablo ortaya koymaktadır."
        )
    else:
        lines.append(
            f"{category_label} kategorisi, {total_reviews} müşteri yorumu analiz edilerek "
            f"değerlendirilmiştir. Ürünlere yönelik eleştiriler ön plana çıkmaktadır."
        )

    # Olumlu aspectler
    if positive:
        aspect_str = ", ".join(f"**{a.name}**" for a in positive[:3])
        lines.append(
            f"Müşteriler özellikle {aspect_str} konularında yüksek memnuniyet bildirmektedir."
        )

    # Olumsuz / dikkat çeken
    if negative:
        neg_str = ", ".join(f"**{a.name}**" for a in negative[:2])
        lines.append(
            f"Öte yandan {neg_str} konusunda bazı olumsuz görüşler dikkat çekmektedir."
        )

    # Karışık
    if mixed:
        mix_str = ", ".join(f"**{a.name}**" for a in mixed[:2])
        lines.append(
            f"{mix_str} konusunda ise görüşler dengeli bir seyir izlemekte, "
            f"hem olumlu hem de olumsuz değerlendirmeler mevcuttur."
        )

    # Kapanış
    if overall_score >= 4.0:
        lines.append(
            f"Genel olarak {category_label} ürünleri müşteri beklentilerini karşılamakta "
            f"ve yüksek oranda tavsiye edilmektedir."
        )
    else:
        lines.append(
            f"Genel olarak {category_label} ürünlerinde müşteri deneyiminin "
            f"iyileştirilebileceğine dair işaretler bulunmaktadır."
        )

    return " ".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# ANA ANALİZ FONKSİYONU
# ─────────────────────────────────────────────────────────────────────────────
def analyze_category(category: str) -> dict[str, Any]:
    print(f"\n[{category}] Yükleniyor...")
    df = load_csv(category)
    print(f"  {len(df)} yorum bulundu.")

    # Dinamik aspect çıkarımı
    aspect_data = extract_dynamic_aspects(df, n_aspects=8)
    print(f"  {len(aspect_data)} aspect keşfedildi.")

    # Her aspect için skor hesapla
    aspect_results: list[AspectResult] = []
    for aspect_name, data in aspect_data.items():
        result = compute_aspect_result(
            df=df,
            aspect_name=aspect_name,
            indices=data["indices"],
            top_terms=data["top_terms"],
        )
        aspect_results.append(result)

    # Yorum sayısına göre sırala
    aspect_results.sort(key=lambda a: a.review_count, reverse=True)

    # Genel skor
    overall_score = float(df["puan"].mean())
    total_reviews = len(df)

    # Özet üret
    category_label = CATEGORY_LABELS.get(category, category)
    summary = generate_summary(category_label, aspect_results, overall_score, total_reviews)

    # Puan dağılımı
    score_dist = {
        str(i): int((df["puan"] == i).sum()) for i in range(1, 6)
    }

    return {
        "category": category,
        "category_label": category_label,
        "total_reviews": total_reviews,
        "overall_score": round(overall_score, 2),
        "overall_sentiment": score_to_sentiment(overall_score),
        "summary": summary,
        "score_distribution": score_dist,
        "aspects": [asdict(a) for a in aspect_results],
        "generated_at": "2026-04-06",
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Dinamik aspect analizi + web JSON üretimi")
    parser.add_argument("--category", type=str, default=None, help="Kategori adı")
    parser.add_argument("--all", action="store_true", help="Tüm kategorileri analiz et")
    parser.add_argument("--output-dir", type=Path, default=WEB_DATA_DIR, help="JSON çıktı klasörü")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    categories = CATEGORIES if args.all else ([args.category] if args.category else ["erkek_kot"])

    all_results = []
    for cat in categories:
        try:
            result = analyze_category(cat)
            all_results.append({
                "category": result["category"],
                "category_label": result["category_label"],
                "total_reviews": result["total_reviews"],
                "overall_score": result["overall_score"],
                "aspect_count": len(result["aspects"]),
            })

            out_path = args.output_dir / f"{cat}.json"
            out_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  ✓ Kaydedildi: {out_path}")
        except Exception as e:
            print(f"  ✗ HATA [{cat}]: {e}")

    # Genel index dosyası
    index_path = args.output_dir / "index.json"
    index_path.write_text(
        json.dumps({"categories": all_results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n✓ Index kaydedildi: {index_path}")


if __name__ == "__main__":
    main()
