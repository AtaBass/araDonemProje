"""
BERTopic ile yorumlardan topic/aspect extraction.

Girdi:
- reviews_output.csv (en az review_text kolonu)

Çıktı:
- topic_results.csv  (yorum + topic)
- topic_info.csv     (topic özet tablosu)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class Config:
    input_csv: Path = Path("reviews_output.csv")
    output_results_csv: Path = Path("topic_results.csv")
    output_info_csv: Path = Path("topic_info.csv")
    output_mapped_csv: Path = Path("topic_results_mapped.csv")
    output_aspect_summary_csv: Path = Path("aspect_summary.csv")
    text_column: str = "review_text"
    min_text_len: int = 5
    embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    top_n_words: int = 10
    min_topic_size: int = 30
    nr_topics: int | str = 12


# Manuel topic -> aspect mapping (guncellenebilir)
# Not: Topic id'ler her egitimde degisebilir; topic_info.csv'ye bakarak revize edin.
ASPECT_NAMES: dict[int, str] = {
    -1: "Genel Memnuniyet / Diger",
    0: "Pisirme Performansi",
    1: "Hediye / Anne",
    2: "Tesekkur / Memnuniyet",
    3: "Fiyat-Performans",
    4: "Tavsiye / Genel Memnuniyet",
    5: "Genel Begeni",
    6: "Kargo ve Paketleme",
    7: "Kullanim Memnuniyeti",
    8: "Kapasite / Aile Uygunlugu",
    9: "Anne Memnuniyeti",
    10: "Indirim / Fiyat",
}


def resolve_input_csv(input_csv: Path) -> Path:
    """Input yoksa preprocess klasöründen ilk uygun CSV'yi seç."""
    if input_csv.exists():
        return input_csv

    fallback_dirs = [Path("../preprocess"), Path("preprocess")]
    for d in fallback_dirs:
        if d.exists():
            candidates = sorted(d.glob("*.csv"))
            if candidates:
                print(f"Uyari: '{input_csv}' bulunamadi. Otomatik secilen dosya: {candidates[0]}")
                return candidates[0]

    raise FileNotFoundError(
        f"Girdi dosyasi bulunamadi: {input_csv}. "
        "Ornek kullanim: python bertopic_aspect_extraction.py --input ../preprocess/airfryer_preprocessed.csv"
    )


def load_reviews(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Girdi dosyasi bulunamadi: {input_csv}")
    return pd.read_csv(input_csv, encoding="utf-8-sig")


def clean_reviews(df: pd.DataFrame, text_column: str, min_text_len: int) -> pd.DataFrame:
    if text_column not in df.columns:
        raise ValueError(f"'{text_column}' kolonu bulunamadı. Mevcut kolonlar: {list(df.columns)}")

    work = df.copy()
    work[text_column] = work[text_column].fillna("").astype(str).str.strip()
    work = work[work[text_column].str.len() >= min_text_len].copy()
    work.reset_index(drop=True, inplace=True)
    return work


def resolve_text_column(df: pd.DataFrame, preferred_col: str) -> str:
    if preferred_col in df.columns:
        return preferred_col
    fallback_cols = ["temiz urun yorum", "urun yorumu", "yorum"]
    for col in fallback_cols:
        if col in df.columns:
            print(f"Uyari: '{preferred_col}' bulunamadi, '{col}' kolonu kullaniliyor.")
            return col
    raise ValueError(f"Metin kolonu bulunamadi. Mevcut kolonlar: {list(df.columns)}")


def build_topic_model(
    embedding_model_name: str,
    top_n_words: int,
    min_topic_size: int,
    nr_topics: int | str,
) -> BERTopic:
    embedding_model = SentenceTransformer(embedding_model_name)
    model = BERTopic(
        embedding_model=embedding_model,
        language="multilingual",
        top_n_words=top_n_words,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        verbose=True,
    )
    return model


def fit_transform_topics(model: BERTopic, texts: list[str]) -> tuple[list[int], list[float]]:
    topics, probs = model.fit_transform(texts)
    return topics, probs


def add_topics_to_dataframe(df: pd.DataFrame, topics: list[int]) -> pd.DataFrame:
    out = df.copy()
    out["topic"] = topics
    return out


def print_readable_topics(model: BERTopic, topic_info: pd.DataFrame) -> None:
    print("\n=== Topic Ozeti (Okunabilir Tablo) ===")
    print(topic_info.to_string(index=False))

    valid_topics = [t for t in topic_info["Topic"].tolist() if t != -1]
    print("\n=== Topic Anahtar Kelimeleri ===")
    for topic_id in valid_topics:
        words_scores = model.get_topic(topic_id) or []
        top_words = [word for word, _ in words_scores[:10]]
        print(f"Topic {topic_id}: {', '.join(top_words)}")


def save_outputs(
    topic_results_df: pd.DataFrame,
    topic_info_df: pd.DataFrame,
    results_path: Path,
    info_path: Path,
) -> None:
    topic_results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    topic_info_df.to_csv(info_path, index=False, encoding="utf-8-sig")


def apply_aspect_mapping(
    topic_results_df: pd.DataFrame,
    topic_info_df: pd.DataFrame,
    aspect_names: dict[int, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped = topic_results_df.copy()
    mapped["aspect"] = mapped["topic"].map(lambda t: aspect_names.get(int(t), "Diger"))

    # Topic info tablosuna da aspect ekle
    info_mapped = topic_info_df.copy()
    info_mapped["aspect"] = info_mapped["Topic"].map(lambda t: aspect_names.get(int(t), "Diger"))
    return mapped, info_mapped


def build_aspect_summary(mapped_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        mapped_df.groupby("aspect", dropna=False)
        .size()
        .reset_index(name="yorum_sayisi")
        .sort_values("yorum_sayisi", ascending=False)
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="BERTopic ile yorumlardan topic extraction")
    parser.add_argument("--input", type=Path, default=Config.input_csv, help="Girdi CSV yolu")
    parser.add_argument("--text-column", type=str, default=Config.text_column, help="Metin kolonu")
    parser.add_argument("--output-results", type=Path, default=Config.output_results_csv, help="Yorum+topic cikti dosyasi")
    parser.add_argument("--output-info", type=Path, default=Config.output_info_csv, help="Topic ozet cikti dosyasi")
    parser.add_argument("--output-mapped", type=Path, default=Config.output_mapped_csv, help="Yorum+topic+aspect cikti dosyasi")
    parser.add_argument("--output-aspect-summary", type=Path, default=Config.output_aspect_summary_csv, help="Aspect ozet cikti dosyasi")
    parser.add_argument("--min-text-len", type=int, default=Config.min_text_len, help="Minimum yorum uzunlugu")
    parser.add_argument("--min-topic-size", type=int, default=Config.min_topic_size, help="Topic olusumu icin minimum yorum sayisi")
    parser.add_argument(
        "--nr-topics",
        type=str,
        default=str(Config.nr_topics),
        help="Hedef topic sayisi (ornek: 8) veya 'auto'",
    )
    args = parser.parse_args()

    cfg = Config(
        input_csv=resolve_input_csv(args.input),
        output_results_csv=args.output_results,
        output_info_csv=args.output_info,
        output_mapped_csv=args.output_mapped,
        output_aspect_summary_csv=args.output_aspect_summary,
        text_column=args.text_column,
        min_text_len=args.min_text_len,
        min_topic_size=args.min_topic_size,
        nr_topics=(args.nr_topics if args.nr_topics == "auto" else int(args.nr_topics)),
    )

    print(f"Girdi dosyasi: {cfg.input_csv}")
    raw_df = load_reviews(cfg.input_csv)
    print(f"Toplam satir: {len(raw_df)}")

    text_col = resolve_text_column(raw_df, cfg.text_column)
    clean_df = clean_reviews(raw_df, text_column=text_col, min_text_len=cfg.min_text_len)
    print(f"Temizleme sonrasi satir: {len(clean_df)}")

    texts = clean_df[text_col].tolist()
    topic_model = build_topic_model(
        cfg.embedding_model_name,
        top_n_words=cfg.top_n_words,
        min_topic_size=cfg.min_topic_size,
        nr_topics=cfg.nr_topics,
    )
    topics, _ = fit_transform_topics(topic_model, texts)

    topic_results_df = add_topics_to_dataframe(clean_df, topics)
    topic_info_df = topic_model.get_topic_info()

    outlier_count = int((topic_results_df["topic"] == -1).sum())
    print(f"Outlier (topic=-1) yorum sayisi: {outlier_count}")

    print_readable_topics(topic_model, topic_info_df)
    print("\n=== topic_model.get_topic_info() ===")
    print(topic_info_df.to_string(index=False))

    save_outputs(
        topic_results_df=topic_results_df,
        topic_info_df=topic_info_df,
        results_path=cfg.output_results_csv,
        info_path=cfg.output_info_csv,
    )

    mapped_results_df, mapped_topic_info_df = apply_aspect_mapping(
        topic_results_df=topic_results_df,
        topic_info_df=topic_info_df,
        aspect_names=ASPECT_NAMES,
    )
    aspect_summary_df = build_aspect_summary(mapped_results_df)

    mapped_results_df.to_csv(cfg.output_mapped_csv, index=False, encoding="utf-8-sig")
    aspect_summary_df.to_csv(cfg.output_aspect_summary_csv, index=False, encoding="utf-8-sig")

    print(f"\nKaydedildi: {cfg.output_results_csv}")
    print(f"Kaydedildi: {cfg.output_info_csv}")
    print(f"Kaydedildi: {cfg.output_mapped_csv}")
    print(f"Kaydedildi: {cfg.output_aspect_summary_csv}")

    # Manuel aspect mapping icin ornek yapi
    aspect_names = ASPECT_NAMES
    print("\nOrnek manual aspect mapping:")
    print(aspect_names)


if __name__ == "__main__":
    main()

