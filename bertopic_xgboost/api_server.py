"""
OpinionMiner — Canlı Analiz API Sunucusu

Çalıştırma:
    cd /Users/atabas/Desktop/aradonem
    .venv/bin/python bertopic_xgboost/api_server.py

Tarayıcıda açın: http://localhost:5001

Not: macOS'ta 5000 portu sıkça AirPlay Receiver tarafından kullanılır.
"""

from __future__ import annotations

import os
# OpenMP / tokenizer thread çakışmalarını önle (macOS Flask ortamında gerekli)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import sys
from pathlib import Path

import requests as _requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Proje kök dizinini ve code/ klasörünü path'e ekle
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "code"))
sys.path.insert(0, str(_THIS_DIR))

from trendyol_scraper import scrape_trendyol  # noqa: E402
from bertopic_product_analyzer import analyze_product_bertopic  # noqa: E402

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3:latest"

# ─── UYGULAMA ──────────────────────────────────────────────────────
app = Flask(__name__, static_folder="web", static_url_path="")
CORS(app)


# ─── OLLAMA ÖZET ───────────────────────────────────────────────────
def ollama_summary(product_name: str, aspects: list, overall_score: float, total_reviews: int) -> str | None:
    """Ollama ile Türkçe özet üret. Başarısız olursa None döner (template'e fallback)."""
    aspect_lines = []
    for a in aspects:
        sentiment = "olumlu" if a["avg_score"] >= 4.2 else ("olumsuz" if a["avg_score"] < 3.2 else "karışık")
        aspect_lines.append(
            f"- {a['name']}: {a['avg_score']:.1f}/5 puan ({sentiment}, {a['review_count']} yorum, "
            f"%{a['positive_pct']} olumlu / %{a['negative_pct']} olumsuz)"
        )

    # Her aspectten en iyi örnek yorumu ekle
    sample_lines = []
    for a in aspects[:4]:
        if a.get("sample_reviews"):
            sample_lines.append(f'"{a["sample_reviews"][0][:120]}"')

    if not aspect_lines:
        return None

    # Örnek yorumları aspect bazında zengin şekilde hazırla
    review_block_lines = []
    for a in aspects[:5]:
        reviews = a.get("sample_reviews", [])[:2]
        sentiment = "olumlu" if a["avg_score"] >= 4.2 else ("olumsuz" if a["avg_score"] < 3.2 else "karışık")
        if reviews:
            review_block_lines.append(f'[{sentiment.upper()} | {a["review_count"]} yorum]')
            for r in reviews:
                review_block_lines.append(f'  • "{r[:130]}"')
    review_block = "\n".join(review_block_lines) if review_block_lines else "\n".join(sample_lines)

    prompt = f"""Sen profesyonel bir Türkçe ürün değerlendirme editörüsün. \
Görevin, gerçek müşteri yorumlarını okuyarak ürün hakkında akıcı ve bilgilendirici bir paragraf yazmaktır.

=== ÜRÜN BİLGİSİ ===
Ürün adı : {product_name[:80]}
Genel puan: {overall_score:.1f} / 5.0  ({total_reviews} müşteri yorumu)

=== MÜŞTERİ YORUMLARINDAN ALINTILAR ===
{review_block}

=== İSTATİSTİKSEL ÖZET ===
{chr(10).join(aspect_lines)}

=== YAZIM GÖREVİ ===
Yukarıdaki GERÇEK müşteri alıntılarına dayanarak 4-5 cümlelik, akıcı bir Türkçe değerlendirme paragrafı yaz.

=== MUTLAKA UYULMASI GEREKEN KURALLAR ===
1. DİL: Yalnızca Türkçe. İngilizce kelime, cümle veya "Translation:" bölümü KESİNLİKLE YASAK.
2. ÇEVİRİ YASAK: Türkçe yazdıktan sonra İngilizce çeviri EKLEME. Tek dil: Türkçe.
3. KAYNAK: Yalnızca alıntılarda geçen konuları yaz. Olmayan özellik UYDURMA.
4. DOĞALLIK: Alıntıları birebir kopyalama; kendi cümlelerinle anlat.
5. BİÇİM: Madde işareti veya tire KULLANMA. Tek düz paragraf.
6. BAŞLANGIÇ: İlk kelime "Bu" olacak. "Here is", "İşte", "Note:" YAZMA.
7. UZUNLUK: 4-5 cümle.
8. ÇIKTI: Sadece Türkçe paragraf. Başka hiçbir şey EKLEME.

Bu"""

    import re as _re

    _TR_CHARS = set("ğüşıöçĞÜŞİÖÇ")
    _EN_WORDS = {
        "the", "and", "that", "with", "some", "only", "but", "for", "are",
        "their", "this", "have", "been", "they", "customers", "noting",
        "overall", "positive", "negative", "product", "feedback", "purchase",
        "satisfied", "quality", "however", "while", "also", "although",
        "here", "sure", "happy", "help", "based", "review", "paragraph",
    }

    def _is_english(sentence: str) -> bool:
        """Cümle İngilizce mi? Türkçe karakter yoksa ve İngilizce kelimeler varsa evet."""
        if any(c in sentence for c in _TR_CHARS):
            return False
        words = set(sentence.lower().split())
        return len(words & _EN_WORDS) >= 2

    def _clean_llm_output(raw: str) -> str:
        text = raw.strip().strip('"').strip()

        # Başta/sonda bilinen İngilizce kalıpları kes
        for marker in ["Translation:", "Note:", "Çeviri:", "English:", "İngilizce:",
                       "I'd be happy", "I'd be happy", "Here's a", "Here is a", "Sure,"]:
            low = text.lower()
            if marker.lower() in low:
                idx = low.index(marker.lower())
                if idx < 100:
                    # Başta → Türkçe cümleleri bul
                    text = " ".join(
                        s.strip() for s in _re.split(r"(?<=[.!?])\s+", text)
                        if s.strip() and not _is_english(s)
                    )
                else:
                    # Sonda → kes
                    text = text[:idx].strip()

        # Cümle bazında İngilizce olanları filtrele (ortadaki geçişleri yakalar)
        sentences = _re.split(r"(?<=[.!?])\s+", text)
        clean_sentences = [s for s in sentences if s.strip() and not _is_english(s)]
        if clean_sentences and len(clean_sentences) < len(sentences):
            text = " ".join(clean_sentences)

        return text.strip('"').strip()

    try:
        resp = _requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "system": "Sen yalnızca Türkçe yazan bir asistansın. Hiçbir zaman İngilizce yazma. Giriş cümlesi veya açıklama yazma. Sadece istenen metni yaz.",
                "prompt": prompt,
                "stream": False,
            },
            timeout=60,
        )
        if resp.status_code == 200:
            text = _clean_llm_output(resp.json().get("response", ""))
            if len(text) > 30:
                return text
    except Exception:
        pass
    return None


# ─── STATİK DOSYALAR ───────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("web", "index.html")


# ─── CANLI ANALİZ API ──────────────────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
def analyze():
    body = request.get_json(force=True, silent=True) or {}
    url = (body.get("url") or "").strip()
    max_pages = int(body.get("max_pages", 5))
    if max_pages <= 0:
        max_pages = None  # scraper'da None = tüm sayfalar

    if not url:
        return jsonify({"error": "URL gerekli"}), 400
    if "trendyol.com" not in url:
        return jsonify({"error": "Geçerli bir Trendyol ürün URL'si girin"}), 400

    # Scraping
    try:
        df = scrape_trendyol(url, max_pages=max_pages, delay=0.8)
    except Exception as exc:
        return jsonify({"error": f"Scraping hatası: {exc}"}), 500

    if df.empty:
        return jsonify({"error": "Yorum bulunamadı. URL'nin doğruluğunu kontrol edin."}), 404

    # Sütun adlarını product_analyzer'ın beklediği formata çevir
    df = df.rename(columns={
        "urun_adi":    "urun adi",
        "urun_url":    "urun url",
        "yorum_temiz": "temiz urun yorum",
    })

    product_url  = df["urun url"].iloc[0]
    product_name = df["urun adi"].iloc[0] or "Trendyol Ürünü"

    # Aspect analizi
    result = analyze_product_bertopic(
        product_df=df,
        product_url=product_url,
        product_name=product_name,
        category="",
        n_topics=6,
    )

    if not result:
        return jsonify({"error": "Yeterli yorum bulunamadı (en az 5 yorum gerekli)."}), 400

    result["category_label"] = "Canlı Analiz"
    result["is_live"] = True

    # Ollama ile AI özet üret; başarısız olursa template özet kalır
    ai_sum = ollama_summary(
        product_name=product_name,
        aspects=result.get("aspects", []),
        overall_score=result["overall_score"],
        total_reviews=result["total_reviews"],
    )
    if ai_sum:
        result["summary"] = ai_sum
        result["summary_source"] = "ollama"
    else:
        result["summary_source"] = "template"

    return jsonify(result)


# ─── YORUM SCRAPE (analiz yok, ham liste döner) ────────────────────
@app.route("/api/scrape-reviews", methods=["POST"])
def scrape_reviews():
    body = request.get_json(force=True, silent=True) or {}
    url = (body.get("url") or "").strip()
    max_reviews = min(int(body.get("max_reviews", 50)), 500)

    if not url:
        return jsonify({"error": "URL gerekli"}), 400
    if "trendyol.com" not in url:
        return jsonify({"error": "Geçerli bir Trendyol ürün URL'si girin"}), 400

    try:
        # max_reviews / 20 sayfa yeterli (sayfa başı 20 yorum)
        import math
        pages = max(1, math.ceil(max_reviews / 20))
        df = scrape_trendyol(url, max_pages=pages, delay=0.8, max_reviews=max_reviews)
    except Exception as exc:
        return jsonify({"error": f"Scraping hatası: {exc}"}), 500

    if df.empty:
        return jsonify({"error": "Yorum bulunamadı."}), 404

    reviews = []
    for _, row in df.iterrows():
        text = str(row.get("yorum", "") or "").strip()
        if len(text) < 5:
            continue
        reviews.append({
            "text":  text,
            "score": int(row.get("puan", 3)),
            "date":  str(row.get("tarih", "")),
            "user":  str(row.get("kullanici", "")),
        })

    product_name = str(df["urun_adi"].iloc[0]) if "urun_adi" in df.columns else "Trendyol Ürünü"
    return jsonify({"reviews": reviews, "product_name": product_name, "total": len(reviews)})


# ─── YORUM BAZLI ANALİZ ────────────────────────────────────────────
@app.route("/api/analyze-reviews", methods=["POST"])
def analyze_reviews():
    import pandas as pd

    body = request.get_json(force=True, silent=True) or {}
    reviews = body.get("reviews", [])
    product_name = (body.get("product_name") or "Yorum Bazlı Analiz").strip()

    if not reviews:
        return jsonify({"error": "En az 1 yorum gerekli"}), 400

    rows = []
    for r in reviews:
        text = str(r.get("text", "")).strip()
        if len(text) < 5:
            continue
        rows.append({
            "urun adi":        product_name,
            "urun url":        "review-test",
            "temiz urun yorum": text,
            "puan":            float(r.get("score", 3)),
        })

    if not rows:
        return jsonify({"error": "Geçerli yorum bulunamadı"}), 400

    df = pd.DataFrame(rows)

    result = analyze_product_bertopic(
        product_df=df,
        product_url="review-test",
        product_name=product_name,
        category="",
        n_topics=6,
    )

    if not result:
        return jsonify({"error": "Yorum metinleri çok kısa veya boş."}), 400

    result["category_label"] = "Yorum Bazlı Test"
    result["is_live"] = True

    ai_sum = ollama_summary(
        product_name=product_name,
        aspects=result.get("aspects", []),
        overall_score=result["overall_score"],
        total_reviews=result["total_reviews"],
    )
    if ai_sum:
        result["summary"] = ai_sum
        result["summary_source"] = "ollama"
    else:
        result["summary_source"] = "template"

    return jsonify(result)


# ─── SUNUCU ────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpinionMiner canlı analiz API sunucusu")
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Dinlenecek port (varsayılan: 5001; macOS'ta 5000 AirPlay için ayrılmış olabilir)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  OpinionMiner — Canlı Analiz Sunucusu")
    print(f"  Tarayıcıda açın: http://localhost:{args.port}")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)
