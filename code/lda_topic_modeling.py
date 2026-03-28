"""
LDA tabanlı Aspect Extraction (multi-label) - preprocess sonrası yorumlar üzerinden.

Amaç:
- preprocess/ klasöründeki her *_preprocessed.csv dosyasını bağımsız analiz et
- LDA ile topic/aspect aday terimlerini çıkar
- Her yoruma TEK TEK bakarak bir veya birden fazla aspect ataması yap (multi-label)

Çıktılar (dosya bazlı):
- aspect_results/lda/{product}_lda_aspects.csv
- aspect_results/lda/{product}_lda_aspect_summary.csv
- aspect_results/lda/{product}_lda_aspect_report.txt

Örnek kullanım:
    python lda_topic_modeling.py --input-dir ../preprocess --output-dir ../aspect_results/lda --n-topics 5 --ngram-range 1,1
    python lda_topic_modeling.py --only airfryer_preprocessed --limit-files 1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import re
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


@dataclass(frozen=True)
class Config:
    INPUT_DIR: Path = Path("../preprocess")
    OUTPUT_DIR: Path = Path("../aspect_results/lda")

    TEXT_COLUMN: str = "temiz urun yorum"

    N_TOPICS: int = 5
    N_TOP_WORDS: int = 10

    MIN_DF: int = 3
    MAX_DF: float = 0.6

    # (1,1) -> unigram, (1,2) -> unigram + bigram
    NGRAM_RANGE: tuple[int, int] = (1, 2)

    RANDOM_STATE: int = 42
    MAX_ITER: int = 20

    # Çok kısa yorumları tekrar ele (preprocess aşaması var ama baseline güvenli olsun)
    MIN_TEXT_LEN: int = 3

    # Yorum için topic seçimi (multi-label aspect için)
    TOPIC_PROB_THRESHOLD: float = 0.10
    TOPIC_TOPK: int = 3

    # Aspect normalize eşlemelerinde çok agresif eşleşmeyi azaltmak için
    # (token içinde geçen aday kelime sayısı altındaysa aspect eklenmez)
    MIN_ASPECT_TOKEN_MATCH: int = 1

    # Raporda topic/top-terms görmek için
    SAVE_DEBUG_TERMS: bool = False


# Aspect normalize eşlemeleri (baseline).
# Not: Stemming/lemmatization yapılmaz; eşleşme için token içinde substring kontrolü kullanılır.
ASPECT_SYNONYMS: dict[str, list[str]] = {
    # performans / deneyim
    "performans": ["hızlı", "çabuk", "performans", "çalışıyor"],
    # bağlantı / eşleşme
    "bağlantı": ["bluetooth", "bağlantı", "eşleş", "eşleşme", "eşle", "bağlan", "bağlantı"],
    # ses
    "ses": ["ses", "hoparlör", "hoparlor", "duyul", "gürültü"],
    # batarya / şarj
    "batarya": ["şarj", "pil", "batarya", "şarja"],
    # kalite / malzeme
    "kalite": ["kalite", "kaliteli", "malzeme", "kumaş", "sağlam", "dayan", "işçilik", "güvenilir"],
    # tasarım / görünüm / estetik
    "tasarım": ["tasarım", "tasarim", "görünüş", "görünüm", "estetik", "şık", "güzel"],
    # renk / görünüm rengi
    "renk": ["renk", "renkli", "renk tonu"],
    # beden/ölçü
    "beden": ["beden", "kalıp", "ölçü", "numara", "ölç"],
    # fiyat
    "fiyat": ["fiyat", "ücret", "pahalı", "indirim", "makul", "bütçe"],
}

# Bazı eş anlamlar birden fazla aspect'e gidebildiği için (örn. "güzel"),
# öncelik sırası ile tek bir aspect'e indirgemeyi kolaylaştırıyoruz.
ASPECT_PRIORITY: list[str] = [
    "batarya",
    "ses",
    "bağlantı",
    "kalite",
    "fiyat",
    "renk",
    "beden",
    "tasarım",
    "performans",
]

_TURKISH_CASE_MAP = str.maketrans("İIĞÜŞÖÇ", "iığüşöç")


def turkish_lower(text: str) -> str:
    if text is None:
        return ""
    return str(text).translate(_TURKISH_CASE_MAP).lower()


def normalize_token_to_aspect(token: str) -> str | None:
    """Tek bir kelime/token'dan aspect ismi döndürür (tek aspect).

    Multi-label için yorum bazında birden fazla token/aspect birikecek.
    """
    t = turkish_lower(token).strip()
    if not t:
        return None

    for aspect in ASPECT_PRIORITY:
        for syn in ASPECT_SYNONYMS.get(aspect, []):
            syn_l = turkish_lower(syn).strip()
            if syn_l and syn_l in t:
                return aspect
    return None


_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


def tokenize_for_aspect(text: str) -> set[str]:
    """LDA'nın CountVectorizer tokenizasyonuna benzer şekilde kelime çıkar."""
    if text is None:
        return set()
    t = turkish_lower(text)
    return set(_TOKEN_RE.findall(t))


def _safe_read_csv(path: Path) -> pd.DataFrame:
    # preprocess çıktıları zaten tek satırda virgül yığılması yapmıyor ama güvenli olsun
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def load_preprocessed_csv(file_path: Path, text_column: str) -> pd.DataFrame:
    df = _safe_read_csv(file_path)
    if text_column not in df.columns:
        # Kolon adı yanlışsa net hata ver
        raise ValueError(f"'{text_column}' kolonu yok: {file_path.name}. Mevcut: {list(df.columns)}")

    # Null / boş yorumları güvenli yönet
    df[text_column] = df[text_column].fillna("").astype(str)
    df[text_column] = df[text_column].map(lambda s: s.strip())

    # Çok kısa yorumları tekrar ele (preprocess aşaması var ama yine de baseline güvenli olsun)
    mask = df[text_column].str.len() >= Config.MIN_TEXT_LEN  # type: ignore[attr-defined]
    df = df.loc[mask].copy()
    return df


def build_vectorizer(
    *,
    min_df: int,
    max_df: float,
    ngram_range: tuple[int, int],
) -> CountVectorizer:
    # Çok agresif filtreleme yok; sadece token sınırı + ngram
    # Türkçe karakterler dahil unicode word'leri alır.
    return CountVectorizer(
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        token_pattern=r"(?u)\b\w\w+\b",  # tek harfleri eleyerek gürültüyü düşürür
        lowercase=False,  # girdiler preprocess sonrası zaten küçük harfe dönüyor
    )


def run_lda(
    texts: list[str],
    *,
    vectorizer: CountVectorizer,
    n_topics: int,
    random_state: int,
    max_iter: int,
) -> tuple[CountVectorizer, LatentDirichletAllocation, np.ndarray]:
    X = vectorizer.fit_transform(texts)

    # Yorum sayısı çok azsa n_topics'u küçült
    n_topics_eff = min(n_topics, max(1, len(texts)))
    lda = LatentDirichletAllocation(
        n_components=n_topics_eff,
        random_state=random_state,
        learning_method="batch",
        max_iter=max_iter,
        evaluate_every=-1,
    )
    lda.fit(X)

    topic_distributions = lda.transform(X)  # (n_docs, n_topics)
    return vectorizer, lda, topic_distributions


def _top_terms_for_topic(
    lda: LatentDirichletAllocation,
    feature_names: np.ndarray,
    topic_id: int,
    n_top_words: int,
) -> list[str]:
    topic_weights = lda.components_[topic_id]
    idx = np.argsort(topic_weights)[::-1][:n_top_words]
    return feature_names[idx].tolist()


def assign_dominant_topics(
    topic_distributions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dominant_topic = np.argmax(topic_distributions, axis=1)
    dominant_topic_score = np.max(topic_distributions, axis=1)
    return dominant_topic, dominant_topic_score


def select_topics_for_comment(
    topic_probs: np.ndarray,
    *,
    prob_threshold: float,
    topk: int,
) -> np.ndarray:
    """Her yorum için birden fazla topic seç (multi-label için)."""
    candidate = np.where(topic_probs >= prob_threshold)[0]
    if candidate.size == 0:
        # Eşik yoksa en yüksek birkaç topic ile devam et
        top_idx = np.argsort(topic_probs)[::-1][: max(1, topk)]
        return top_idx

    if candidate.size <= topk:
        return candidate

    # Fazla topic varsa topk olacak şekilde daralt
    sorted_candidate = candidate[np.argsort(topic_probs[candidate])[::-1]]
    return sorted_candidate[:topk]


def extract_topic_term_aspects(
    *,
    lda: LatentDirichletAllocation,
    feature_names: np.ndarray,
    n_topics: int,
    n_top_words: int,
) -> list[list[tuple[str, set[str]]]]:
    """Topic başına top-term -> normalize aspect eşlemesi çıkar."""
    topic_terms_mapped: list[list[tuple[str, set[str]]]] = []

    for topic_id in range(n_topics):
        top_terms = _top_terms_for_topic(lda, feature_names, topic_id, n_top_words=n_top_words)
        mapped_list: list[tuple[str, set[str]]] = []
        for term in top_terms:
            parts = term.split()
            aspects: set[str] = set()
            if len(parts) == 1:
                a = normalize_token_to_aspect(parts[0])
                if a:
                    aspects.add(a)
            else:
                for p in parts:
                    a = normalize_token_to_aspect(p)
                    if a:
                        aspects.add(a)

            if aspects:
                mapped_list.append((term, aspects))

        topic_terms_mapped.append(mapped_list)

    return topic_terms_mapped


def assign_aspects_for_comment(
    *,
    comment_tokens: set[str],
    topic_probs: np.ndarray,
    topic_terms_mapped: list[list[tuple[str, set[str]]]],
    prob_threshold: float,
    topk: int,
    min_aspect_token_match: int,
) -> list[str]:
    """Tek bir yoruma birden fazla aspect atar (multi-label)."""
    selected_topics = select_topics_for_comment(topic_probs, prob_threshold=prob_threshold, topk=topk)

    aspect_counts: dict[str, int] = {}

    for topic_id in selected_topics:
        for term, aspects in topic_terms_mapped[topic_id]:
            term_parts = term.split()
            matched = False
            if len(term_parts) == 1:
                matched = term_parts[0] in comment_tokens
            else:
                matched = all(p in comment_tokens for p in term_parts)

            if not matched:
                continue

            for aspect in aspects:
                aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1

    # Minimum eşik
    matched_aspects = [a for a, c in aspect_counts.items() if c >= min_aspect_token_match]
    # Aspect order: priority listesine göre deterministik sırala
    matched_aspects.sort(key=lambda x: ASPECT_PRIORITY.index(x) if x in ASPECT_PRIORITY else 10_000)
    return matched_aspects


def _parse_stem_for_product(stem: str) -> str:
    # airfryer_preprocessed -> airfryer ; daha okunabilir dosya adı
    return re.sub(r"_preprocessed$", "", stem)


def save_aspect_outputs(
    *,
    input_df: pd.DataFrame,
    input_csv_path: Path,
    text_column: str,
    lda: LatentDirichletAllocation,
    vectorizer: CountVectorizer,
    topic_distributions: np.ndarray,
    topic_terms_mapped: list[list[tuple[str, set[str]]]],
    output_dir: Path,
    n_top_words: int,
    prob_threshold: float,
    topk: int,
    min_aspect_token_match: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = vectorizer.get_feature_names_out()
    n_topics_eff = topic_distributions.shape[1]
    product_prefix = _parse_stem_for_product(input_csv_path.stem)

    aspects_csv_path = output_dir / f"{product_prefix}_lda_aspects.csv"
    summary_csv_path = output_dir / f"{product_prefix}_lda_aspect_summary.csv"
    report_path = output_dir / f"{product_prefix}_lda_report.txt"

    # Yorum bazlı aspect ataması
    comments = input_df[text_column].fillna("").astype(str).map(lambda s: s.strip()).tolist()
    all_tokens = [tokenize_for_aspect(t) for t in comments]

    bulunan_aspectler_list: list[str] = []
    aspect_sayisi: list[int] = []

    # Summary için sayım
    aspect_comment_counts: dict[str, int] = {}  # ilgili_yorum_sayisi
    aspect_freq_counts: dict[str, int] = {}  # frekans (multi-label: yorumda aspect başına 1)

    for i in range(len(comments)):
        aspects = assign_aspects_for_comment(
            comment_tokens=all_tokens[i],
            topic_probs=topic_distributions[i],
            topic_terms_mapped=topic_terms_mapped,
            prob_threshold=prob_threshold,
            topk=topk,
            min_aspect_token_match=min_aspect_token_match,
        )
        if not aspects:
            bulunan_aspectler_list.append("[]")
            aspect_sayisi.append(0)
            continue

        # Stringified list formatı (Excel/CSV uyumlu)
        bulunan_aspectler_list.append("[" + ", ".join(aspects) + "]")
        aspect_sayisi.append(len(aspects))

        # Summary sayıları
        for a in aspects:
            aspect_comment_counts[a] = aspect_comment_counts.get(a, 0) + 1
            aspect_freq_counts[a] = aspect_freq_counts.get(a, 0) + 1

    # Yorum bazlı çıktı CSV
    cols_needed = ["urun adi", "urun url", text_column, "puan", "tarih"]
    for c in cols_needed:
        if c not in input_df.columns:
            raise ValueError(f"Gerekli kolon '{c}' yok: {input_csv_path.name}. Mevcut: {list(input_df.columns)}")

    out_df = input_df[cols_needed].copy()
    out_df = out_df.rename(columns={text_column: "temiz urun yorum"})
    out_df["bulunan_aspectler"] = bulunan_aspectler_list
    out_df["aspect_sayisi"] = aspect_sayisi

    out_df = out_df[["urun adi", "urun url", "temiz urun yorum", "puan", "tarih", "bulunan_aspectler", "aspect_sayisi"]]
    out_df.to_csv(aspects_csv_path, index=False, encoding="utf-8-sig", lineterminator="\n")

    # Aspect summary CSV
    summary_rows = []
    aspects_sorted = sorted(aspect_freq_counts.keys(), key=lambda a: aspect_freq_counts[a], reverse=True)
    total_docs = len(input_df)
    for a in aspects_sorted:
        freq = int(aspect_freq_counts.get(a, 0))
        related = int(aspect_comment_counts.get(a, 0))
        # oran bilgisi raporda kullanılabilir; dosyada sadece istenen kolonlar
        summary_rows.append({"aspect": a, "frekans": freq, "ilgili_yorum_sayisi": related})

    summary_df = pd.DataFrame(summary_rows, columns=["aspect", "frekans", "ilgili_yorum_sayisi"])
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig", lineterminator="\n")

    # Rapor
    lines: list[str] = []
    lines.append(f"Girdi: {input_csv_path.name}")
    lines.append(f"Toplam yorum: {len(input_df)}")
    lines.append(f"Topic sayısı (effective): {n_topics_eff}")
    lines.append(f"Topic seçimi: prob>={prob_threshold} veya topk={topk}")
    lines.append("")
    lines.append("Topic -> top terms (baseline) + aspect adayları:")
    lines.append("")
    for topic_id in range(n_topics_eff):
        top_terms = _top_terms_for_topic(lda, feature_names, topic_id, n_top_words=n_top_words)
        mapped_aspects: set[str] = set()
        for term, aspects in topic_terms_mapped[topic_id]:
            mapped_aspects.update(aspects)
        mapped_aspects_str = ", ".join(sorted(mapped_aspects))
        lines.append(f"- topic {topic_id}: {', '.join(top_terms)} | mapped aspects: {mapped_aspects_str}")

    lines.append("")
    lines.append("Aspect dağılımı:")
    for row in summary_rows[:50]:
        lines.append(f"- {row['aspect']}: frekans={row['frekans']} | ilgili_yorum={row['ilgili_yorum_sayisi']}")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "aspects_csv_path": aspects_csv_path,
        "summary_csv_path": summary_csv_path,
        "report_path": report_path,
        "n_topics_eff": n_topics_eff,
        "n_aspects": len(summary_rows),
        "n_comments": len(input_df),
    }


def save_lda_outputs(
    *,
    input_df: pd.DataFrame,
    input_csv_path: Path,
    vectorizer: CountVectorizer,
    lda: LatentDirichletAllocation,
    topic_distributions: np.ndarray,
    output_dir: Path,
    n_top_words: int,
    save_topic_dists: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = vectorizer.get_feature_names_out()
    dominant_topic, dominant_topic_score = assign_dominant_topics(topic_distributions)

    product_prefix = _parse_stem_for_product(input_csv_path.stem)
    topics_csv_path = output_dir / f"{product_prefix}_lda_topics.csv"
    summary_csv_path = output_dir / f"{product_prefix}_lda_topic_summary.csv"
    report_path = output_dir / f"{product_prefix}_lda_report.txt"

    # 1) Yorum bazlı çıktı
    cols_needed = ["urun adi", "urun url", "temiz urun yorum", "puan", "tarih"]
    for c in cols_needed:
        if c not in input_df.columns:
            raise ValueError(f"Gerekli kolon '{c}' yok: {input_csv_path.name}. Mevcut: {list(input_df.columns)}")

    out_topics = input_df[cols_needed].copy()
    out_topics["dominant_topic"] = dominant_topic.astype(int).astype(str)
    out_topics["dominant_topic_score"] = np.round(dominant_topic_score, 6)

    if save_topic_dists:
        # İsteğe bağlı: her topic için olasılık kolonları
        for k in range(topic_distributions.shape[1]):
            out_topics[f"topic_{k}_score"] = np.round(topic_distributions[:, k], 6)

    out_topics.to_csv(topics_csv_path, index=False, encoding="utf-8-sig", lineterminator="\n")

    # 2) Topic özet
    n_topics_eff = topic_distributions.shape[1]
    total_docs = len(input_df)
    dominant_counts = pd.Series(dominant_topic).value_counts().sort_index()

    topic_summaries = []
    for topic_id in range(n_topics_eff):
        top_terms = _top_terms_for_topic(lda, feature_names, topic_id, n_top_words=n_top_words)
        yorum_sayisi = int(dominant_counts.get(topic_id, 0))
        oran = (yorum_sayisi / total_docs * 100.0) if total_docs else 0.0
        topic_summaries.append(
            {
                "topic_id": str(topic_id),
                "top_terms": ", ".join(top_terms),
                "yorum_sayisi": yorum_sayisi,
                "oran": np.round(oran, 3),
            }
        )

    out_summary = pd.DataFrame(topic_summaries)
    out_summary.to_csv(summary_csv_path, index=False, encoding="utf-8-sig", lineterminator="\n")

    # 3) Report
    lines: list[str] = []
    lines.append(f"Girdi: {input_csv_path.name}")
    lines.append(f"Toplam yorum (kullanılan): {total_docs}")
    lines.append(f"Topic sayısı (effective): {n_topics_eff}")
    lines.append("")
    lines.append("Topic özetleri:")

    for row in topic_summaries:
        lines.append(
            f"- topic {row['topic_id']}: "
            f"{row['top_terms']} | "
            f"yorum_sayisi={row['yorum_sayisi']} | oran={row['oran']}%"
        )

    lines.append("")
    lines.append("Kısa değerlendirme (baseline):")
    lines.append("Bu rapor sklearn LDA ile üretilmiş başlangıç/deneme amaçlı bir sonuçtur.")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    # Debug/metadata
    return {
        "topics_csv_path": topics_csv_path,
        "summary_csv_path": summary_csv_path,
        "report_path": report_path,
        "used_docs": total_docs,
        "effective_n_topics": n_topics_eff,
    }


def process_all_files_for_lda(
    input_dir: Path,
    output_dir: Path,
    *,
    text_column: str,
    n_topics: int,
    n_top_words: int,
    min_df: int,
    max_df: float,
    ngram_range: tuple[int, int],
    random_state: int,
    max_iter: int,
    save_topic_dists: bool,
    only: str | None,
    limit_files: int | None,
) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input klasörü yok: {input_dir}")

    csv_files = sorted(input_dir.glob("*.csv"))
    if only:
        csv_files = [p for p in csv_files if only.lower() in p.stem.lower()]
    if limit_files is not None:
        csv_files = csv_files[: max(0, limit_files)]

    if not csv_files:
        print(f"Uyarı: {input_dir} içinde analiz edilecek CSV bulunamadı.")
        return

    print(f"Analiz edilecek dosya sayısı: {len(csv_files)}")

    for idx, csv_path in enumerate(csv_files, start=1):
        print(f"\n[{idx}/{len(csv_files)}] İşleniyor: {csv_path.name}")

        df = _safe_read_csv(csv_path)
        if text_column not in df.columns:
            raise ValueError(f"'{text_column}' kolonu yok: {csv_path.name}. Mevcut: {list(df.columns)}")

        df[text_column] = df[text_column].fillna("").astype(str).map(lambda s: s.strip())
        df = df.loc[df[text_column].str.len() >= Config.MIN_TEXT_LEN]  # type: ignore[attr-defined]

        used_texts = df[text_column].tolist()
        print(f"  Kullanılan yorum: {len(used_texts)}")
        if not used_texts:
            print("  Uyarı: Boş yorum → atlandı.")
            continue

        vectorizer = build_vectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range)
        vectorizer, lda, topic_distributions = run_lda(
            used_texts,
            vectorizer=vectorizer,
            n_topics=n_topics,
            random_state=random_state,
            max_iter=max_iter,
        )

        result_meta = save_lda_outputs(
            input_df=df,
            input_csv_path=csv_path,
            vectorizer=vectorizer,
            lda=lda,
            topic_distributions=topic_distributions,
            output_dir=output_dir,
            n_top_words=n_top_words,
            save_topic_dists=save_topic_dists,
        )

        print(f"  Topic sayısı: {result_meta['effective_n_topics']}")
        print(f"  Çıktılar: {result_meta['topics_csv_path'].name}, {result_meta['summary_csv_path'].name}, report üretildi.")


def process_all_files_for_aspect_extraction(
    input_dir: Path,
    output_dir: Path,
    *,
    text_column: str,
    n_topics: int,
    n_top_words: int,
    min_df: int,
    max_df: float,
    ngram_range: tuple[int, int],
    random_state: int,
    max_iter: int,
    topic_prob_threshold: float,
    topic_topk: int,
    min_aspect_token_match: int,
    only: str | None,
    limit_files: int | None,
) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input klasörü yok: {input_dir}")

    csv_files = sorted(input_dir.glob("*.csv"))
    if only:
        csv_files = [p for p in csv_files if only.lower() in p.stem.lower()]
    if limit_files is not None:
        csv_files = csv_files[: max(0, limit_files)]

    if not csv_files:
        print(f"Uyarı: {input_dir} içinde analiz edilecek CSV bulunamadı.")
        return

    print(f"Analiz edilecek dosya sayısı: {len(csv_files)}")

    for idx, csv_path in enumerate(csv_files, start=1):
        print(f"\n[{idx}/{len(csv_files)}] İşleniyor: {csv_path.name}")

        df = _safe_read_csv(csv_path)
        if text_column not in df.columns:
            raise ValueError(f"'{text_column}' kolonu yok: {csv_path.name}. Mevcut: {list(df.columns)}")

        df[text_column] = df[text_column].fillna("").astype(str).map(lambda s: s.strip())
        df = df.loc[df[text_column].str.len() >= Config.MIN_TEXT_LEN].copy()  # type: ignore[attr-defined]

        used_texts = df[text_column].tolist()
        print(f"  Kullanılan yorum: {len(used_texts)}")
        if not used_texts:
            print("  Uyarı: Boş yorum → atlandı.")
            continue

        # LDA için metni tutarlı hale getir
        used_texts = [turkish_lower(t) for t in used_texts]

        vectorizer = build_vectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range)
        vectorizer, lda, topic_distributions = run_lda(
            used_texts,
            vectorizer=vectorizer,
            n_topics=n_topics,
            random_state=random_state,
            max_iter=max_iter,
        )

        n_topics_eff = topic_distributions.shape[1]
        feature_names = vectorizer.get_feature_names_out()
        topic_terms_mapped = extract_topic_term_aspects(
            lda=lda,
            feature_names=feature_names,
            n_topics=n_topics_eff,
            n_top_words=n_top_words,
        )

        result_meta = save_aspect_outputs(
            input_df=df,
            input_csv_path=csv_path,
            text_column=text_column,
            lda=lda,
            vectorizer=vectorizer,
            topic_distributions=topic_distributions,
            topic_terms_mapped=topic_terms_mapped,
            output_dir=output_dir,
            n_top_words=n_top_words,
            prob_threshold=topic_prob_threshold,
            topk=topic_topk,
            min_aspect_token_match=min_aspect_token_match,
        )

        print(f"  Çıktılar: {result_meta['aspects_csv_path'].name}, {result_meta['summary_csv_path'].name}, report üretildi.")
        print(f"  Topic sayısı (effective): {result_meta['n_topics_eff']} | Aspect sayısı: {result_meta['n_aspects']}")


def _parse_ngram_range(s: str) -> tuple[int, int]:
    # "1,1" veya "1 2" gibi gelebilir
    parts = re.split(r"[,\s]+", s.strip())
    parts = [p for p in parts if p]
    if len(parts) != 2:
        raise ValueError(f"ngram_range formatı geçersiz: '{s}'. Örn: '1,2'")
    return int(parts[0]), int(parts[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess yorumları üzerinde LDA tabanlı aspect extraction (multi-label, dosya bazlı).")
    parser.add_argument("--input-dir", type=Path, default=Config.INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=Config.OUTPUT_DIR)
    parser.add_argument("--text-column", type=str, default=Config.TEXT_COLUMN)

    parser.add_argument("--n-topics", type=int, default=Config.N_TOPICS)
    parser.add_argument("--n-top-words", type=int, default=Config.N_TOP_WORDS)

    parser.add_argument("--min-df", type=int, default=Config.MIN_DF)
    parser.add_argument("--max-df", type=float, default=Config.MAX_DF)
    parser.add_argument("--ngram-range", type=str, default="1,2")

    parser.add_argument("--random-state", type=int, default=Config.RANDOM_STATE)
    parser.add_argument("--max-iter", type=int, default=Config.MAX_ITER)

    parser.add_argument("--topic-prob-threshold", type=float, default=Config.TOPIC_PROB_THRESHOLD)
    parser.add_argument("--topic-topk", type=int, default=Config.TOPIC_TOPK)
    parser.add_argument("--min-aspect-token-match", type=int, default=Config.MIN_ASPECT_TOKEN_MATCH)

    parser.add_argument("--only", type=str, default=None, help="Sadece stem içinde geçen substring için analiz yap")
    parser.add_argument("--limit-files", type=int, default=None, help="Test için max kaç dosya")

    args = parser.parse_args()

    ngram_range = _parse_ngram_range(args.ngram_range)

    process_all_files_for_aspect_extraction(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        text_column=args.text_column,
        n_topics=args.n_topics,
        n_top_words=args.n_top_words,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range=ngram_range,
        random_state=args.random_state,
        max_iter=args.max_iter,
        topic_prob_threshold=args.topic_prob_threshold,
        topic_topk=args.topic_topk,
        min_aspect_token_match=args.min_aspect_token_match,
        only=args.only,
        limit_files=args.limit_files,
    )


if __name__ == "__main__":
    main()

