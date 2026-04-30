"""
product_analyzer.py — Her ürün (URL) için dinamik aspect çıkarımı + puanlama + özet.

Kullanım:
    python3 product_analyzer.py --all          # tüm kategoriler, tüm ürünler
    python3 product_analyzer.py --category airfryer
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# ─── YOLLAR ────────────────────────────────────────────────────────
_THIS_DIR    = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
PREPROCESS_DIR = _PROJECT_ROOT / "preprocess"
WEB_DATA_DIR   = _THIS_DIR / "web" / "data" / "products"

CATEGORIES = [
    "airfryer", "akilli_saat", "erkek_kot", "kadin_canta",
    "kadin_tayt", "kulaklık", "mouse", "nevresim", "sandalye", "spor_ayakkabi",
]
CATEGORY_LABELS = {
    "airfryer": "Airfryer", "akilli_saat": "Akıllı Saat",
    "erkek_kot": "Erkek Kot Pantolon", "kadin_canta": "Kadın Çanta",
    "kadin_tayt": "Kadın Tayt", "kulaklık": "Kablosuz Kulaklık",
    "mouse": "Mouse", "nevresim": "Nevresim Takımı",
    "sandalye": "Sandalye", "spor_ayakkabi": "Spor Ayakkabı",
}

# ─── TÜRKÇE ────────────────────────────────────────────────────────
_TR_MAP = str.maketrans("İIĞÜŞÖÇ", "iığüşöç")
def tr_lower(s: str) -> str:
    return str(s).translate(_TR_MAP).lower()

_EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001FAFF\U00002702-\U000027B0\U0000200D\U0000FE0F]+",
    flags=re.UNICODE,
)
def clean(s: str) -> str:
    s = _EMOJI_RE.sub(" ", str(s))
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\d+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

STOP_WORDS = {
    "bir","bu","da","de","için","ile","ve","ama","çok","en","ben","sen",
    "biz","siz","onlar","mi","mu","mı","var","yok","ki","ne","olan","oldu",
    "olur","gibi","daha","hem","ya","ise","bile","kadar","göre","nasıl",
    "sadece","herkes","hiç","artık","aldım","aldim","aldik","geldi","geldik",
    "beğendim","begendim","begendik","teşekkür","tesekkur","memnun","iyi",
    "güzel","guzel","tam","beden","alın","alabilirsiniz","tavsiye","ederim",
    "herkese","kesinlikle","gerçekten","sipariş","ürün","urun",
}

# ─── ASPECT SEEDS ──────────────────────────────────────────────────
ASPECT_SEEDS: dict[str, list[str]] = {
    "Kumaş & Malzeme":      ["kumaş","kumasi","malzeme","dokus","dokusu","likralı","esnek","yumuşak","kalın","ince","plastik","metal","içerik"],
    "Kalite & Dayanıklılık":["kalite","kaliteli","sağlam","dayanıklı","işçilik","mükemmel","bozuldu","dayandı","ömür"],
    "Beden & Kalıp":        ["beden","kalıp","ölçü","numara","dar","bol","geniş","tam","uydu","oturdu","büyük","küçük"],
    "Fiyat & Değer":        ["fiyat","fiyatı","ücret","ucuz","pahalı","uygun","ekonomik","değer","performans","para"],
    "Kargo & Teslimat":     ["kargo","teslimat","teslim","hızlı","paketleme","paket","ulaştı","gün","süre"],
    "Renk & Görünüm":       ["renk","rengi","renkli","görünüm","gorsel","şık","estetik","tasarım","güzel","siyah","beyaz"],
    "Kullanım Kolaylığı":   ["rahat","kolay","kullanım","kullanımı","pratik","hafif","ergonomik","işlevsel"],
    "Satıcı & Hizmet":      ["satıcı","satici","hizmet","mağaza","firma","hediye","iade","müşteri"],
    # Ürüne özel
    "Pişirme Performansı":  ["pişirme","pişiriyor","kızarmış","lezzetli","çıtır","yağsız","fritozu","ısınma","sıcak"],
    "Ses & Gürültü":        ["ses","gürültü","sessiz","gürültülü","motor","fan"],
    "Batarya & Şarj":       ["batarya","pil","şarj","şarja","şarj süresi","güç","mah"],
    "Bağlantı":             ["bluetooth","wifi","bağlantı","eşleştirme","sinyal","kablosuz"],
    "Ekran & Arayüz":       ["ekran","display","tuş","buton","arayüz","menü","gösterge"],
    "Temizlik & Hijyen":    ["temizlik","temizleme","yıkama","hijyen","koku","çıkarılabilir"],
    "Kapasite & Boyut":     ["kapasite","litre","büyüklük","küçük","büyük","geniş","dar","boyut"],
}

ASPECT_ICONS = {
    "Kumaş":  "🧵", "Malzeme": "🧵",
    "Kalite": "⭐", "Dayanıklılık": "🔩",
    "Beden":  "📏", "Kalıp": "📏",
    "Fiyat":  "💰", "Değer": "💰",
    "Kargo":  "📦", "Teslimat": "📦",
    "Renk":   "🎨", "Görünüm": "🎨",
    "Kullanım":"😌", "Rahat": "😌",
    "Satıcı": "🏪", "Hizmet": "🏪",
    "Pişirme":"🍳", "Ses": "🔊",
    "Batarya":"🔋", "Bağlantı": "📡",
    "Ekran":  "📱", "Temizlik": "🧹",
    "Kapasite":"📦",
}

def get_icon(name: str) -> str:
    for k, v in ASPECT_ICONS.items():
        if k.lower() in name.lower():
            return v
    return "💬"

# ─── YARDIMCILAR ───────────────────────────────────────────────────
def url_to_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]

def score_to_sentiment(s: float) -> str:
    return "positive" if s >= 4.5 else ("neutral" if s >= 3.0 else "negative")

def score_to_label(s: float) -> str:
    if s >= 4.5: return "Olumlu"
    if s >= 3.8: return "Genel Olumlu"
    if s >= 2.5: return "Karışık"
    return "Olumsuz"

def stars(s: float) -> str:
    f = int(s)
    h = "½" if s - f >= 0.4 else ""
    return "★" * f + h + "☆" * (5 - f - (1 if h else 0))

# ─── DİNAMİK ASPECT ÇIKARIM ────────────────────────────────────────
def extract_aspects(df: pd.DataFrame, n_clusters: int = 6) -> dict[str, dict]:
    texts = [tr_lower(clean(t)) for t in df["temiz urun yorum"].tolist()]
    if len(texts) < 5:
        return {}

    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        token_pattern=r"(?u)\b[a-züğışöçıİĞÜŞÖÇ]{3,}\b",
        sublinear_tf=True,
    )
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        return {}

    feature_names = vectorizer.get_feature_names_out()
    n_clust = min(n_clusters, max(2, len(texts) // 4))

    km = KMeans(n_clusters=n_clust, random_state=42, n_init=10, max_iter=200)
    labels = km.fit_predict(X)

    # Her küme → top terms
    cluster_terms: dict[int, list[str]] = {}
    for cid in range(n_clust):
        center = km.cluster_centers_[cid]
        top_idx = np.argsort(center)[::-1][:20]
        terms = [feature_names[i] for i in top_idx if feature_names[i] not in STOP_WORDS and len(feature_names[i]) > 3]
        cluster_terms[cid] = terms[:10]

    # Küme → aspect ismi (seed eşleştirme)
    cluster_aspect: dict[int, str] = {}
    used: set[str] = set()
    for cid, terms in cluster_terms.items():
        best, best_score = None, 0
        for asp, seeds in ASPECT_SEEDS.items():
            score = sum(1 for t in terms if any(s in t or t in s for s in seeds))
            if score > best_score:
                best_score, best = score, asp
        if best and best not in used and best_score >= 1:
            cluster_aspect[cid] = best
            used.add(best)
        else:
            cluster_aspect[cid] = (terms[0].capitalize() if terms else f"Konu {cid+1}")

    # Aspect → indices
    aspect_indices: dict[str, list[int]] = defaultdict(list)
    for idx, cid in enumerate(labels):
        aspect_indices[cluster_aspect[cid]].append(idx)

    # En az 3 yorum olan aspectler
    result: dict[str, dict] = {}
    for cid, asp_name in cluster_aspect.items():
        idxs = aspect_indices[asp_name]
        if len(idxs) < 3:
            continue
        result[asp_name] = {
            "indices": idxs,
            "top_terms": cluster_terms[cid][:8],
        }
    return result

# ─── PUANLAMA ──────────────────────────────────────────────────────
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

def compute_aspect(df: pd.DataFrame, name: str, idxs: list[int], terms: list[str]) -> AspectResult:
    sub = df.iloc[idxs]
    scores = sub["puan"].values
    avg = float(np.mean(scores))
    pos = float(np.mean(scores >= 4.5))
    neg = float(np.mean(scores <= 2.0))
    neu = max(0.0, 1.0 - pos - neg)

    sample = (
        sub["temiz urun yorum"]
        .sort_values(key=lambda x: x.str.len(), ascending=False)
        .head(3)
        .tolist()
    )
    return AspectResult(
        name=name,
        review_count=len(idxs),
        avg_score=round(avg, 2),
        sentiment=score_to_sentiment(avg),
        sentiment_label=score_to_label(avg),
        top_terms=terms,
        positive_pct=round(pos * 100, 1),
        negative_pct=round(neg * 100, 1),
        neutral_pct=round(neu * 100, 1),
        sample_reviews=sample,
        icon=get_icon(name),
    )

# ─── TÜRKÇE ÖZET ───────────────────────────────────────────────────
def generate_summary(
    product_name: str,
    aspects: list[AspectResult],
    overall: float,
    n_reviews: int,
) -> str:
    pos_asp = [a for a in aspects if a.avg_score >= 4.2]
    neg_asp = [a for a in aspects if a.avg_score < 3.2]
    mix_asp = [a for a in aspects if 3.2 <= a.avg_score < 4.2]

    short_name = product_name.split(" - ")[0][:60]
    lines: list[str] = []

    if overall >= 4.5:
        lines.append(
            f"**{short_name}** ürünü için {n_reviews} müşteri yorumu analiz edilmiştir. "
            f"Ürün, genel itibarıyla müşteriler tarafından son derece olumlu karşılanmaktadır."
        )
    elif overall >= 4.0:
        lines.append(
            f"**{short_name}** için toplamda {n_reviews} yorum incelenmiştir. "
            f"Ürün, büyük çoğunluğu tarafından beğenilmektedir."
        )
    elif overall >= 3.0:
        lines.append(
            f"**{short_name}** için {n_reviews} yorum incelenmiştir. "
            f"Müşteri görüşleri karışık bir tablo ortaya koymaktadır."
        )
    else:
        lines.append(
            f"**{short_name}** için {n_reviews} yorum incelenmiştir. "
            f"Ürüne ilişkin eleştiriler öne çıkmaktadır."
        )

    if pos_asp:
        asp_str = ", ".join(f"**{a.name}**" for a in pos_asp[:3])
        lines.append(
            f"Müşteriler {asp_str} konularında yüksek memnuniyet bildirmektedir."
        )

    if neg_asp:
        neg_str = ", ".join(f"**{a.name}**" for a in neg_asp[:2])
        lines.append(
            f"Öte yandan {neg_str} konusunda olumsuz görüşler dikkat çekmektedir."
        )

    if mix_asp:
        mix_str = ", ".join(f"**{a.name}**" for a in mix_asp[:2])
        lines.append(
            f"{mix_str} başlıklarında ise görüşler dengelidir."
        )

    if overall >= 4.0:
        lines.append(
            "Genel olarak ürün, kullanıcı beklentilerini karşılamakta ve yüksek oranda tavsiye edilmektedir."
        )
    else:
        lines.append(
            "Genel değerlendirmede ürünün bazı alanlarda iyileştirmeye ihtiyaç duyduğu görülmektedir."
        )

    return " ".join(lines)

# ─── ÜRÜN ANALİZİ ──────────────────────────────────────────────────
def analyze_product(
    product_df: pd.DataFrame,
    product_url: str,
    product_name: str,
    category: str,
    n_clusters: int = 6,
) -> dict[str, Any]:
    df = product_df[product_df["urun url"] == product_url].copy()
    df["puan"] = pd.to_numeric(df["puan"], errors="coerce").fillna(3.0)
    df = df[df["temiz urun yorum"].str.len() >= 8].copy()
    df.reset_index(drop=True, inplace=True)

    n_reviews = len(df)
    if n_reviews == 0:
        return {}

    overall = float(df["puan"].mean())

    aspect_data = extract_aspects(df, n_clusters=n_clusters)
    aspect_results: list[AspectResult] = []
    for asp_name, data in aspect_data.items():
        r = compute_aspect(df, asp_name, data["indices"], data["top_terms"])
        aspect_results.append(r)

    aspect_results.sort(key=lambda a: a.review_count, reverse=True)

    summary = generate_summary(product_name, aspect_results, overall, n_reviews)

    score_dist = {str(i): int((df["puan"] == i).sum()) for i in range(1, 6)}

    # Örnek yorumlar (en uzun 5)
    top_reviews = (
        df["temiz urun yorum"]
        .sort_values(key=lambda x: x.str.len(), ascending=False)
        .head(5)
        .tolist()
    )

    return {
        "product_id": url_to_id(product_url),
        "product_name": product_name,
        "product_url": product_url,
        "category": category,
        "category_label": CATEGORY_LABELS.get(category, category),
        "total_reviews": n_reviews,
        "overall_score": round(overall, 2),
        "overall_sentiment": score_to_sentiment(overall),
        "stars": stars(overall),
        "summary": summary,
        "score_distribution": score_dist,
        "aspects": [asdict(a) for a in aspect_results],
        "top_reviews": top_reviews,
        "generated_at": "2026-04-06",
    }

# ─── KATEGORİ ANALİZİ ──────────────────────────────────────────────
def analyze_category(category: str, output_dir: Path, n_clusters: int = 6) -> list[dict]:
    csv_path = PREPROCESS_DIR / f"{category}_preprocessed.csv"
    if not csv_path.exists():
        print(f"  ⚠ Dosya yok: {csv_path}")
        return []

    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df["puan"] = pd.to_numeric(df["puan"], errors="coerce").fillna(3.0)
    df["temiz urun yorum"] = df["temiz urun yorum"].fillna("").str.strip()

    cat_dir = output_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    products = df.groupby("urun url")["temiz urun yorum"].count().reset_index()
    products.columns = ["urun url", "count"]
    products = products.sort_values("count", ascending=False)

    index_list = []
    for _, row in products.iterrows():
        url = row["urun url"]
        name = df[df["urun url"] == url]["urun adi"].iloc[0] if "urun adi" in df.columns else url

        result = analyze_product(df, url, name, category, n_clusters=n_clusters)
        if not result:
            continue

        pid = result["product_id"]
        out_path = cat_dir / f"{pid}.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        index_list.append({
            "product_id": pid,
            "product_name": name[:80],
            "product_url": url,
            "total_reviews": result["total_reviews"],
            "overall_score": result["overall_score"],
            "overall_sentiment": result["overall_sentiment"],
            "aspect_count": len(result["aspects"]),
        })
        print(f"    ✓ {name[:50][:50]} ({result['total_reviews']} yorum, {len(result['aspects'])} aspect)")

    # Kategori index
    idx_path = cat_dir / "index.json"
    idx_path.write_text(json.dumps({"category": category, "products": index_list}, ensure_ascii=False, indent=2), encoding="utf-8")
    return index_list

# ─── CLI ───────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=WEB_DATA_DIR)
    parser.add_argument("--n-clusters", type=int, default=6)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cats = CATEGORIES if args.all else ([args.category] if args.category else ["erkek_kot"])

    global_index = {}
    for cat in cats:
        print(f"\n[{cat}] analiz ediliyor...")
        idx = analyze_category(cat, args.output_dir, n_clusters=args.n_clusters)
        global_index[cat] = {
            "category_label": CATEGORY_LABELS.get(cat, cat),
            "product_count": len(idx),
            "products": idx,
        }

    g_path = args.output_dir / "global_index.json"
    g_path.write_text(json.dumps(global_index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ Global index: {g_path}")

if __name__ == "__main__":
    main()
