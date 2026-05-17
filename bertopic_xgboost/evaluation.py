"""
evaluation.py — BERTopic + XGBoost pipeline değerlendirmesi

Otomatik Metrikler:
  1. Kümeleme Kalitesi  → Silhouette Score
  2. Topic Coherence    → C_V skoru (Gensim)
  3. XGBoost F1/Acc     → Train/test split ile sınıflandırma başarısı
  4. Özet Kalitesi      → ROUGE-1, ROUGE-L

İnsan Değerlendirmesi:
  → human_eval_form.json + web/human_eval.html üretir
    (aspect anlamlılık + özet kalitesi anketi)

Kullanım:
    python3 evaluation.py --category mouse
    python3 evaluation.py --all
"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
from xgboost import XGBClassifier
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from rouge_score import rouge_scorer as rouge_lib

# Aynı modülden yardımcı fonksiyonlar
from bertopic_product_analyzer import (
    tr_lower, clean, STOP_WORDS, _NOISE,
    label_from_keywords, build_bertopic_model,
    PREPROCESS_DIR, CATEGORIES,
)

_THIS_DIR  = Path(__file__).resolve().parent
EVAL_DIR   = _THIS_DIR / "evaluation_output"
EVAL_DIR.mkdir(exist_ok=True)

ROUGE = rouge_lib.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)

# ─── YARDIMCILAR ──────────────────────────────────────────────────────────────

def load_category(category: str) -> pd.DataFrame:
    files = list(PREPROCESS_DIR.glob(f"{category}_preprocessed.csv"))
    if not files:
        raise FileNotFoundError(f"{category} CSV bulunamadı")
    df = pd.read_csv(files[0])
    df = df.dropna(subset=["temiz urun yorum", "puan"])
    df["temiz urun yorum"] = df["temiz urun yorum"].astype(str)
    df = df[df["temiz urun yorum"].str.len() >= 5]
    return df.reset_index(drop=True)


def rouge_summary_vs_reviews(summary: str, reviews: list[str]) -> dict:
    """Özeti gerçek yorumlarla ROUGE ile karşılaştır."""
    if not reviews or not summary:
        return {"rouge1": 0.0, "rougeL": 0.0}
    scores_r1, scores_rl = [], []
    for rev in reviews[:20]:
        s = ROUGE.score(summary, rev)
        scores_r1.append(s["rouge1"].fmeasure)
        scores_rl.append(s["rougeL"].fmeasure)
    return {
        "rouge1": round(float(np.mean(scores_r1)), 4),
        "rougeL": round(float(np.mean(scores_rl)), 4),
    }


def coherence_score(topic_words: list[list[str]], texts: list[list[str]]) -> float:
    """Gensim C_V coherence — topic kelimelerinin metinlerle tutarlılığı."""
    try:
        dictionary = Dictionary(texts)
        cm = CoherenceModel(
            topics=topic_words,
            texts=texts,
            dictionary=dictionary,
            coherence="c_v",
        )
        return round(float(cm.get_coherence()), 4)
    except Exception:
        return float("nan")


# ─── ANA DEĞERLENDİRME ────────────────────────────────────────────────────────

def evaluate_category(category: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  DEĞERLENDİRME: {category.upper()}")
    print(f"{'='*60}")

    df = load_category(category)
    all_texts = [tr_lower(clean(t)) for t in df["temiz urun yorum"].tolist()]

    # ── Embedding ──
    print("  [1/4] Embedding üretiliyor...")
    st_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = st_model.encode(all_texts, show_progress_bar=False, batch_size=64)

    # ── BERTopic ──
    print("  [2/4] BERTopic eğitiliyor...")
    model = build_bertopic_model(all_texts, n_topics=12)
    topics, _ = model.fit_transform(all_texts, embeddings=embeddings)
    df["topic"] = topics

    valid_mask   = np.array(topics) != -1
    outlier_mask = ~valid_mask
    n_outliers   = int(outlier_mask.sum())
    n_total      = len(topics)
    valid_tids   = [t for t in set(topics) if t != -1]
    n_topics_found = len(valid_tids)

    print(f"     {n_topics_found} topic, {n_outliers}/{n_total} outlier ({100*n_outliers/n_total:.1f}%)")

    # ── 1. Silhouette Score ──
    sil_score = float("nan")
    if valid_mask.sum() >= 10 and len(set(np.array(topics)[valid_mask])) >= 2:
        sil_score = silhouette_score(
            embeddings[valid_mask],
            np.array(topics)[valid_mask],
            metric="cosine",
            sample_size=min(2000, int(valid_mask.sum())),
            random_state=42,
        )
        sil_score = round(float(sil_score), 4)
    print(f"     Silhouette Score: {sil_score:.4f}")

    # ── 2. Topic Coherence ──
    print("  [3/4] Coherence hesaplanıyor...")
    topic_info   = model.get_topic_info()
    topic_words_list: list[list[str]] = []
    raw_labels: dict[int, str] = {}
    raw_terms:  dict[int, list[str]] = {}

    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            continue
        word_score_pairs = model.get_topic(tid)
        words  = [w for w, _ in word_score_pairs if w not in STOP_WORDS and len(w) > 3]
        scores = [s for w, s in word_score_pairs if w not in STOP_WORDS and len(w) > 3]
        if not words:
            continue
        top_words = words[:10]
        topic_words_list.append(top_words)
        label = label_from_keywords(words, scores)
        if label:
            raw_labels[tid] = label
            raw_terms[tid]  = words[:10]

    tokenized = [t.split() for t in all_texts]
    cv_score  = coherence_score(topic_words_list, tokenized) if topic_words_list else float("nan")
    print(f"     Topic Coherence (C_V): {cv_score:.4f}" if not np.isnan(cv_score) else "     Topic Coherence: hesaplanamadı")

    # ── 3. XGBoost F1 / Accuracy ──
    print("  [4/4] XGBoost değerlendiriliyor...")
    xgb_metrics: dict = {}
    if valid_mask.sum() >= 40 and len(valid_tids) >= 2:
        X = embeddings[valid_mask]
        y = np.array(topics)[valid_mask]
        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, stratify=y_enc,
        )

        xgb = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, verbosity=0,
        )
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)

        acc  = round(float(accuracy_score(y_test, y_pred)), 4)
        f1_w = round(float(f1_score(y_test, y_pred, average="weighted")), 4)
        f1_m = round(float(f1_score(y_test, y_pred, average="macro")), 4)
        xgb_metrics = {
            "accuracy":        acc,
            "f1_weighted":     f1_w,
            "f1_macro":        f1_m,
            "n_train":         len(X_train),
            "n_test":          len(X_test),
            "n_classes":       len(le.classes_),
        }
        print(f"     Accuracy: {acc:.4f}  |  F1 (weighted): {f1_w:.4f}  |  F1 (macro): {f1_m:.4f}")
    else:
        print("     Yeterli veri yok — XGBoost atlandı")

    # ── 4. ROUGE (JSON'lardan özet okuyarak) ──
    json_dir = _THIS_DIR / "web" / "data" / "products" / category
    rouge_scores: list[dict] = []
    if json_dir.exists():
        for jf in list(json_dir.glob("*.json"))[:20]:
            try:
                d = json.loads(jf.read_text())
                summary = d.get("summary", "")
                aspects = d.get("aspects", [])
                if not summary or not aspects:
                    continue
                all_reviews = []
                for asp in aspects:
                    all_reviews += [r["text"] for r in asp.get("all_reviews", [])]
                rs = rouge_summary_vs_reviews(summary, all_reviews)
                rs["product"] = d.get("product_name", "")[:40]
                rouge_scores.append(rs)
            except Exception:
                continue

    avg_rouge1 = round(float(np.mean([r["rouge1"] for r in rouge_scores])), 4) if rouge_scores else float("nan")
    avg_rougeL = round(float(np.mean([r["rougeL"] for r in rouge_scores])), 4) if rouge_scores else float("nan")
    print(f"     ROUGE-1: {avg_rouge1:.4f}  |  ROUGE-L: {avg_rougeL:.4f}")

    return {
        "category":          category,
        "n_reviews":         n_total,
        "n_topics":          n_topics_found,
        "outlier_count":     n_outliers,
        "outlier_pct":       round(100 * n_outliers / n_total, 2),
        "silhouette":        sil_score,
        "coherence_cv":      cv_score,
        "xgboost":           xgb_metrics,
        "rouge": {
            "rouge1_avg":    avg_rouge1,
            "rougeL_avg":    avg_rougeL,
            "n_products":    len(rouge_scores),
        },
        "topic_labels":      raw_labels,
        "topic_terms":       raw_terms,
    }


# ─── İNSAN DEĞERLENDİRME FORMU ────────────────────────────────────────────────

def build_human_eval_form(results: list[dict]) -> None:
    """Seçilmiş aspect + özet örneklerinden insan değerlendirme anketi üret."""
    random.seed(42)
    form_items: list[dict] = []

    for res in results:
        cat  = res["category"]
        jdir = _THIS_DIR / "web" / "data" / "products" / cat
        if not jdir.exists():
            continue

        files = list(jdir.glob("*.json"))
        random.shuffle(files)

        aspect_count = 0
        summary_count = 0

        for jf in files:
            if aspect_count >= 5 and summary_count >= 3:
                break
            try:
                d = json.loads(jf.read_text())
                aspects = d.get("aspects", [])
                summary = d.get("summary", "")
                pname   = d.get("product_name", "")[:60]

                # Aspect anlamlılık sorular (kategori başına max 5)
                for asp in aspects[:3]:
                    if aspect_count >= 5:
                        break
                    reviews = [r["text"] for r in asp.get("all_reviews", [])[:5]]
                    if not reviews:
                        continue
                    form_items.append({
                        "type":        "aspect",
                        "category":    cat,
                        "product":     pname,
                        "aspect_name": asp.get("name", ""),
                        "top_terms":   asp.get("top_terms", [])[:5],
                        "reviews":     reviews,
                        "question":    "Bu aspect başlığı bu yorumları ne kadar iyi temsil ediyor?",
                        "scale":       "1 (Hiç uygun değil) — 5 (Çok uygun)",
                        "score":       None,
                        "comment":     "",
                    })
                    aspect_count += 1

                # Özet kalite soruları (kategori başına max 3)
                if summary and summary_count < 3:
                    all_reviews = []
                    for asp in aspects:
                        all_reviews += [r["text"] for r in asp.get("all_reviews", [])[:3]]
                    form_items.append({
                        "type":        "summary",
                        "category":    cat,
                        "product":     pname,
                        "summary":     summary,
                        "reviews":     all_reviews[:8],
                        "questions": {
                            "dogruluk":  {"q": "Özette yazılanlar yorumlarda gerçekten var mı? (Doğruluk)", "score": None},
                            "akicilik":  {"q": "Türkçe doğal ve akıcı mı? (Dil kalitesi)", "score": None},
                            "kapsam":    {"q": "Önemli bir konu atlanmış mı? (Kapsam)", "score": None},
                        },
                        "scale":       "1 (Çok kötü) — 5 (Çok iyi)",
                        "comment":     "",
                    })
                    summary_count += 1

            except Exception:
                continue

    # JSON kaydet
    form_path = EVAL_DIR / "human_eval_form.json"
    form_path.write_text(json.dumps(form_items, ensure_ascii=False, indent=2))
    print(f"\n  İnsan değerlendirme formu: {form_path} ({len(form_items)} soru)")

    # HTML arayüzü üret
    _write_human_eval_html(form_items)


def _write_human_eval_html(items: list[dict]) -> None:
    aspect_items  = [i for i in items if i["type"] == "aspect"]
    summary_items = [i for i in items if i["type"] == "summary"]

    def aspect_card(item: dict, idx: int) -> str:
        reviews_html = "".join(
            f'<li style="margin:4px 0;color:#444">{r[:150]}{"…" if len(r)>150 else ""}</li>'
            for r in item["reviews"]
        )
        terms = ", ".join(f"<b>{t}</b>" for t in item["top_terms"])
        return f"""
        <div class="card" id="a{idx}">
          <div class="cat-badge">{item['category']}</div>
          <div class="product-name">{item['product']}</div>
          <div class="aspect-name">🏷 Aspect: <strong>{item['aspect_name']}</strong></div>
          <div class="terms">Anahtar terimler: {terms}</div>
          <div class="reviews-title">Örnek Yorumlar:</div>
          <ul class="reviews">{reviews_html}</ul>
          <div class="question">{item['question']}</div>
          <div class="scale-label">{item['scale']}</div>
          <div class="stars">
            {"".join(f'<label><input type="radio" name="a{idx}" value="{v}"> {v}</label>' for v in range(1,6))}
          </div>
          <textarea class="comment" placeholder="Opsiyonel yorum..." data-id="a{idx}"></textarea>
        </div>"""

    def summary_card(item: dict, idx: int) -> str:
        reviews_html = "".join(
            f'<li style="margin:4px 0;color:#444">{r[:150]}{"…" if len(r)>150 else ""}</li>'
            for r in item["reviews"]
        )
        summary_escaped = item['summary'].replace('\n', '<br>')
        qs_html = "".join(
            f"""<div class="subq">
              <div class="question">{qdata['q']}</div>
              <div class="stars">
                {"".join(f'<label><input type="radio" name="s{idx}_{qk}" value="{v}"> {v}</label>' for v in range(1,6))}
              </div>
            </div>"""
            for qk, qdata in item["questions"].items()
        )
        return f"""
        <div class="card" id="s{idx}">
          <div class="cat-badge">{item['category']}</div>
          <div class="product-name">{item['product']}</div>
          <div class="reviews-title">Orijinal Yorumlar (örnek):</div>
          <ul class="reviews">{reviews_html}</ul>
          <div class="reviews-title" style="margin-top:12px">Üretilen Özet:</div>
          <div class="summary-box">{summary_escaped}</div>
          <div class="scale-label" style="margin-top:8px">{item['scale']}</div>
          {qs_html}
          <textarea class="comment" placeholder="Opsiyonel yorum..." data-id="s{idx}"></textarea>
        </div>"""

    aspect_html  = "".join(aspect_card(it, i) for i, it in enumerate(aspect_items))
    summary_html = "".join(summary_card(it, i) for i, it in enumerate(summary_items))

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<title>İnsan Değerlendirme Formu</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #f4f6fa; color: #222; }}
  h1 {{ text-align:center; padding: 32px 16px 8px; font-size: 1.6rem; color: #1a1a2e; }}
  h2 {{ text-align:center; color: #666; font-weight:400; font-size:1rem; margin-bottom:24px; }}
  .section-title {{ max-width:860px; margin: 32px auto 12px; font-size:1.2rem;
                    font-weight:700; color:#1a1a2e; padding: 0 16px; }}
  .card {{ background:#fff; border-radius:12px; padding:20px 24px; margin: 0 auto 18px;
           max-width:860px; box-shadow:0 2px 8px rgba(0,0,0,.08); }}
  .cat-badge {{ display:inline-block; background:#e8f0fe; color:#1967d2;
                border-radius:20px; padding:2px 10px; font-size:.78rem; margin-bottom:8px; }}
  .product-name {{ font-weight:600; font-size:.95rem; color:#333; margin-bottom:6px; }}
  .aspect-name {{ font-size:1.05rem; margin-bottom:4px; }}
  .terms {{ font-size:.85rem; color:#666; margin-bottom:10px; }}
  .reviews-title {{ font-weight:600; font-size:.88rem; color:#555; margin-bottom:6px; }}
  .reviews {{ padding-left:18px; font-size:.85rem; margin-bottom:14px; }}
  .summary-box {{ background:#f8f9ff; border-left:4px solid #4a6cf7; padding:12px 16px;
                  border-radius:6px; font-size:.9rem; line-height:1.7; margin-bottom:14px; }}
  .question {{ font-weight:600; font-size:.95rem; margin-bottom:4px; }}
  .scale-label {{ font-size:.82rem; color:#888; margin-bottom:8px; }}
  .stars label {{ margin-right:16px; font-size:.95rem; cursor:pointer; }}
  .stars input {{ margin-right:4px; accent-color:#4a6cf7; }}
  .subq {{ border-top: 1px solid #eee; padding-top:10px; margin-top:10px; }}
  .comment {{ width:100%; margin-top:12px; padding:8px 10px; border:1px solid #ddd;
              border-radius:6px; font-family:inherit; font-size:.85rem; resize:vertical;
              min-height:52px; }}
  .submit-btn {{ display:block; margin: 24px auto 48px; padding:14px 48px;
                 background:#4a6cf7; color:#fff; border:none; border-radius:8px;
                 font-size:1rem; font-weight:600; cursor:pointer; }}
  .submit-btn:hover {{ background:#3558d6; }}
  #result-msg {{ text-align:center; font-weight:600; color:#2e7d32; display:none;
                 margin-bottom:24px; font-size:1.05rem; }}
</style>
</head>
<body>
<h1>İnsan Değerlendirme Formu</h1>
<h2>Dinamik Aspect Çıkarımı ve Özet Kalitesi Değerlendirmesi</h2>

<div class="section-title">Bölüm 1 — Aspect Anlamlılık Değerlendirmesi</div>
{aspect_html}

<div class="section-title">Bölüm 2 — Özet Kalitesi Değerlendirmesi</div>
{summary_html}

<button class="submit-btn" onclick="submitForm()">Değerlendirmeyi Kaydet</button>
<div id="result-msg">✓ Değerlendirme kaydedildi: human_eval_results.json</div>

<script>
function submitForm() {{
  const results = {{ aspects: [], summaries: [] }};

  document.querySelectorAll('[id^="a"]').forEach((card, i) => {{
    const sel = card.querySelector('input[type=radio]:checked');
    results.aspects.push({{
      id: card.id,
      score: sel ? parseInt(sel.value) : null,
      comment: card.querySelector('textarea').value,
    }});
  }});

  document.querySelectorAll('[id^="s"]').forEach((card, i) => {{
    const scores = {{}};
    ['dogruluk','akicilik','kapsam'].forEach(k => {{
      const sel = card.querySelector(`input[name="${{card.id}}_${{k}}"]:checked`);
      scores[k] = sel ? parseInt(sel.value) : null;
    }});
    results.summaries.push({{
      id: card.id,
      scores: scores,
      comment: card.querySelector('textarea').value,
    }});
  }});

  const blob = new Blob([JSON.stringify(results, null, 2)], {{type:'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'human_eval_results.json';
  a.click();
  document.getElementById('result-msg').style.display = 'block';
}}
</script>
</body>
</html>"""

    html_path = EVAL_DIR / "human_eval.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"  İnsan değerlendirme arayüzü : {html_path}")


# ─── RAPOR YAZICI ──────────────────────────────────────────────────────────────

def print_report(results: list[dict]) -> None:
    print(f"\n{'='*60}")
    print("  OTOMATİK DEĞERLENDİRME RAPORU")
    print(f"{'='*60}")
    print(f"{'Kategori':<16} {'Topics':>6} {'Outlier%':>9} {'Silhouette':>11} {'Coherence':>10} {'XGB F1-w':>9} {'ROUGE-1':>8}")
    print("-" * 75)

    sils, cohers, f1s, r1s = [], [], [], []
    for r in results:
        sil  = r.get("silhouette", float("nan"))
        coh  = r.get("coherence_cv", float("nan"))
        f1w  = r.get("xgboost", {}).get("f1_weighted", float("nan"))
        r1   = r.get("rouge", {}).get("rouge1_avg", float("nan"))
        out  = r.get("outlier_pct", 0)

        def fmt(v): return f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "   N/A"
        print(f"{r['category']:<16} {r['n_topics']:>6} {out:>8.1f}% {fmt(sil):>11} {fmt(coh):>10} {fmt(f1w):>9} {fmt(r1):>8}")

        if not np.isnan(sil):  sils.append(sil)
        if not np.isnan(coh):  cohers.append(coh)
        if not np.isnan(f1w):  f1s.append(f1w)
        if not np.isnan(r1):   r1s.append(r1)

    print("-" * 75)
    def avg(lst): return f"{np.mean(lst):.4f}" if lst else "N/A"
    print(f"{'ORTALAMA':<16} {'':>6} {'':>9} {avg(sils):>11} {avg(cohers):>10} {avg(f1s):>9} {avg(r1s):>8}")

    # JSON raporu kaydet
    report_path = EVAL_DIR / "auto_eval_report.json"
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n  Detaylı rapor kaydedildi: {report_path}")


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", action="append", dest="categories")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    cats = CATEGORIES if args.all else (args.categories or ["mouse"])

    results = []
    for cat in cats:
        try:
            r = evaluate_category(cat)
            results.append(r)
        except Exception as e:
            print(f"  ✗ {cat}: {e}")

    if results:
        print_report(results)
        build_human_eval_form(results)


if __name__ == "__main__":
    main()
