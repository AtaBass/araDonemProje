"""
Topic/Aspect etiketli yorumlar için sentiment + aspect analizi.

Girdi:
- topic_results_mapped.csv

Cikti:
- reviews_with_sentiment.csv
- aspect_sentiment_summary.csv
- final_sentiment_output.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass(frozen=True)
class Config:
    input_csv: Path = Path("topic_results_mapped.csv")
    output_reviews_csv: Path = Path("reviews_with_sentiment.csv")
    output_aspect_summary_csv: Path = Path("aspect_sentiment_summary.csv")
    output_final_json: Path = Path("final_sentiment_output.json")

    # Metin kolonu fallback sırası
    text_column_candidates: tuple[str, ...] = (
        "review_text",
        "temiz urun yorum",
        "urun yorumu",
        "yorum",
    )
    # Aspect kolonu fallback (kullanıcı typo'su için acpect de dahil)
    aspect_column_candidates: tuple[str, ...] = ("aspect", "acpect")

    # 3 sinifli sentiment modeli (multilingual, Turkce yorumlarda da kullanilabilir)
    sentiment_model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    batch_size: int = 32
    max_length: int = 256

    min_text_len: int = 3


# Model label -> standard sentiment mapping
# Kolay degistirilebilir, modele gore guncelleyin.
LABEL_MAPPING: dict[str, str] = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    "negative": "negative",
    "negatif": "negative",
    "neutral": "neutral",
    "notr": "neutral",
    "positive": "positive",
    "pozitif": "positive",
}


def resolve_column(df: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"{label} kolonu bulunamadi. Mevcut kolonlar: {list(df.columns)}")


def load_and_clean(input_csv: Path, cfg: Config) -> tuple[pd.DataFrame, str, str]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Girdi dosyasi bulunamadi: {input_csv}")

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    text_col = resolve_column(df, cfg.text_column_candidates, "Metin")
    aspect_col = resolve_column(df, cfg.aspect_column_candidates, "Aspect")

    work = df.copy()
    work[text_col] = work[text_col].fillna("").astype(str).str.strip()
    work[aspect_col] = work[aspect_col].fillna("").astype(str).str.strip()
    work.loc[work[aspect_col] == "", aspect_col] = "Diğer"
    work = work[work[text_col].str.len() >= cfg.min_text_len].copy()
    work.reset_index(drop=True, inplace=True)
    return work, text_col, aspect_col


def normalize_sentiment_label(label: str, mapping: dict[str, str]) -> str:
    raw = str(label).strip()
    if raw in mapping:
        return mapping[raw]
    lower = raw.lower()
    if lower in mapping:
        return mapping[lower]
    return "neutral"


def load_model_and_tokenizer(model_name: str) -> tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    # use_fast=False ile sentencepiece/tiktoken fallback kaynakli parser hatalarini azalt
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def run_sentiment_bert(
    texts: list[str],
    *,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    batch_size: int,
    max_length: int,
    label_mapping: dict[str, str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    id2label = model.config.id2label or {}

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            confs, preds = torch.max(probs, dim=-1)

        for pred_id, conf in zip(preds.tolist(), confs.tolist()):
            raw_label = id2label.get(pred_id, f"LABEL_{pred_id}")
            sent = normalize_sentiment_label(raw_label, label_mapping)
            results.append({"label": sent, "score": float(conf)})

    return results


def add_sentiment_columns(
    df: pd.DataFrame,
    text_col: str,
    model_name: str,
    batch_size: int,
    max_length: int,
    label_mapping: dict[str, str],
) -> pd.DataFrame:
    texts = df[text_col].tolist()
    tokenizer, model, device = load_model_and_tokenizer(model_name)
    preds = run_sentiment_bert(
        texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        label_mapping=label_mapping,
    )

    out = df.copy()
    out["sentiment"] = [str(p.get("label", "neutral")).lower() for p in preds]
    out["score"] = [float(p.get("score", 0.0)) for p in preds]
    return out


def compute_aspect_distribution(df: pd.DataFrame, aspect_col: str) -> pd.DataFrame:
    aspect_dist = (
        df.groupby(aspect_col)["sentiment"]
        .value_counts(normalize=True)
        .unstack()
        .fillna(0.0)
    )
    for c in ("positive", "negative", "neutral"):
        if c not in aspect_dist.columns:
            aspect_dist[c] = 0.0
    aspect_dist = aspect_dist[["positive", "negative", "neutral"]].reset_index()
    return aspect_dist


def compute_overall_distribution(df: pd.DataFrame) -> dict[str, float]:
    s = df["sentiment"].value_counts(normalize=True)
    result = {
        "positive": float(s.get("positive", 0.0)),
        "negative": float(s.get("negative", 0.0)),
        "neutral": float(s.get("neutral", 0.0)),
    }
    return result


def summarize_aspect(aspect: str, pos: float, neg: float, neu: float) -> str:
    if neu >= 0.50:
        return f"{aspect} hakkinda yorumlarin buyuk kismi notr; net bir olumlu/olumsuz egilim zayif."
    if neg > pos + 0.10:
        return f"{aspect} kullanicilar tarafindan genelde olumsuz degerlendiriliyor."
    if pos > neg + 0.10:
        if neu >= 0.35:
            return f"{aspect} genel olarak begeniliyor; ancak kayda deger bir notr gorus de var."
        return f"{aspect} kullanicilar tarafindan genelde begeniliyor."
    if abs(pos - neg) <= 0.10:
        return f"{aspect} hakkinda kullanici gorusleri karisik."
    return f"{aspect} icin gorusler dengeli fakat net bir egilim olusmamis."


def build_aspect_output(aspect_dist: pd.DataFrame, aspect_col: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in aspect_dist.iterrows():
        aspect = str(row[aspect_col])
        pos = float(row["positive"])
        neg = float(row["negative"])
        neu = float(row["neutral"])
        records.append(
            {
                "aspect": aspect,
                "sentiment": {
                    "positive": pos,
                    "negative": neg,
                    "neutral": neu,
                },
                "summary": summarize_aspect(aspect, pos, neg, neu),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Aspect bazli sentiment analizi")
    parser.add_argument("--input", type=Path, default=Config.input_csv, help="Girdi CSV")
    parser.add_argument("--output-reviews", type=Path, default=Config.output_reviews_csv, help="Yorum + sentiment cikti CSV")
    parser.add_argument("--output-aspect-summary", type=Path, default=Config.output_aspect_summary_csv, help="Aspect sentiment ozet CSV")
    parser.add_argument("--output-json", type=Path, default=Config.output_final_json, help="Final JSON cikti")
    parser.add_argument("--model", type=str, default=Config.sentiment_model_name, help="Hugging Face model adi")
    parser.add_argument("--batch-size", type=int, default=Config.batch_size, help="Batch boyutu")
    parser.add_argument("--max-length", type=int, default=Config.max_length, help="Tokenizer truncation uzunlugu")
    parser.add_argument("--min-text-len", type=int, default=Config.min_text_len, help="Minimum yorum uzunlugu")
    args = parser.parse_args()

    cfg = Config(
        input_csv=args.input,
        output_reviews_csv=args.output_reviews,
        output_aspect_summary_csv=args.output_aspect_summary,
        output_final_json=args.output_json,
        sentiment_model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        min_text_len=args.min_text_len,
    )

    print(f"Girdi: {cfg.input_csv}")
    df, text_col, aspect_col = load_and_clean(cfg.input_csv, cfg)
    print(f"Kullanilan metin kolonu: {text_col}")
    print(f"Kullanilan aspect kolonu: {aspect_col}")
    print(f"Temizleme sonrasi yorum sayisi: {len(df)}")

    reviews_with_sent = add_sentiment_columns(
        df=df,
        text_col=text_col,
        model_name=cfg.sentiment_model_name,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        label_mapping=LABEL_MAPPING,
    )

    reviews_with_sent.to_csv(cfg.output_reviews_csv, index=False, encoding="utf-8-sig")
    print(f"Kaydedildi: {cfg.output_reviews_csv}")

    aspect_dist = compute_aspect_distribution(reviews_with_sent, aspect_col=aspect_col)
    aspect_dist.to_csv(cfg.output_aspect_summary_csv, index=False, encoding="utf-8-sig")
    print(f"Kaydedildi: {cfg.output_aspect_summary_csv}")

    overall = compute_overall_distribution(reviews_with_sent)
    print("\nGenel sentiment dagilimi:")
    print(json.dumps(overall, ensure_ascii=False, indent=2))

    aspect_records = build_aspect_output(aspect_dist, aspect_col=aspect_col)
    output = {
        "overall_sentiment": overall,
        "aspects": aspect_records,
    }

    cfg.output_final_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Kaydedildi: {cfg.output_final_json}")


if __name__ == "__main__":
    main()

