"""
Sentiment/aspect JSON ciktisini dogal dilde final rapora donusturur.

Girdi:
- final_output.json (veya --input ile farkli dosya)

Cikti:
- natural_final_output.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Config:
    input_json: Path = Path("final_output.json")
    output_json: Path = Path("natural_final_output.json")

    # Genel ve zayif aspectleri filtrelemek icin degistirilebilir liste
    IGNORE_ASPECTS: tuple[str, ...] = (
        "Genel Begeni",
        "Tesekkur / Memnuniyet",
        "Tavsiye / Genel Memnuniyet",
        "Genel Memnuniyet / Diger",
    )

    # Kural esikleri
    strong_gap: float = 0.15
    mixed_gap: float = 0.08
    neutral_high: float = 0.45
    negative_concern: float = 0.25
    positive_strong: float = 0.70


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Girdi JSON bulunamadi: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def format_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def get_sentiment_label(dist: dict[str, float], cfg: Config) -> str:
    p = float(dist.get("positive", 0.0))
    n = float(dist.get("negative", 0.0))
    u = float(dist.get("neutral", 0.0))

    if u >= cfg.neutral_high and u >= p and u >= n:
        return "çoğunlukla nötr"
    if p - n >= cfg.strong_gap:
        return "çoğunlukla pozitif"
    if n - p >= cfg.strong_gap:
        return "çoğunlukla negatif"
    return "karışık"


def build_aspect_summary(aspect: str, dist: dict[str, float], cfg: Config) -> str:
    p = float(dist.get("positive", 0.0))
    n = float(dist.get("negative", 0.0))
    u = float(dist.get("neutral", 0.0))

    if u >= cfg.neutral_high and u >= p and u >= n:
        return f"{aspect} hakkında yorumlar daha çok nötr veya bilgi verici düzeyde."
    if p >= cfg.positive_strong and p - n >= cfg.mixed_gap:
        return f"{aspect} kullanıcılar tarafından belirgin şekilde beğeniliyor."
    if n >= cfg.negative_concern and n - p >= cfg.mixed_gap:
        return f"{aspect} kullanıcılar tarafından en çok eleştirilen konulardan biri."
    if abs(p - n) <= cfg.mixed_gap:
        return f"{aspect} hakkında kullanıcı görüşleri karışık."
    if p > n:
        return f"{aspect} tarafında olumlu görüşler olumsuzlara göre daha baskın."
    return f"{aspect} tarafında olumsuz geri bildirimler daha görünür."


def filter_aspects(aspects: list[dict[str, Any]], ignore_aspects: tuple[str, ...]) -> list[dict[str, Any]]:
    ignore_set = {x.strip().lower() for x in ignore_aspects}
    kept: list[dict[str, Any]] = []
    for a in aspects:
        name = str(a.get("aspect", "")).strip()
        if not name:
            continue
        if name.lower() in ignore_set:
            continue
        kept.append(a)
    return kept


def build_general_comment(aspects: list[dict[str, Any]], cfg: Config) -> str:
    if not aspects:
        return "Filtre sonrası anlamlı bir aspect bulunamadığı için genel değerlendirme sınırlı kaldı."

    scored: list[tuple[str, float, float, float]] = []
    for a in aspects:
        name = str(a.get("aspect", ""))
        d = a.get("sentiment", {}) or {}
        p = float(d.get("positive", 0.0))
        n = float(d.get("negative", 0.0))
        u = float(d.get("neutral", 0.0))
        scored.append((name, p, n, u))

    # En pozitif: positive-negative farkina gore
    pos_sorted = sorted(scored, key=lambda x: (x[1] - x[2], x[1]), reverse=True)
    top_pos = [x[0] for x in pos_sorted if (x[1] - x[2]) >= cfg.mixed_gap][:3]

    # En negatif: negative-positive farkina gore
    neg_sorted = sorted(scored, key=lambda x: (x[2] - x[1], x[2]), reverse=True)
    top_neg = [x[0] for x in neg_sorted if (x[2] - x[1]) >= cfg.mixed_gap and x[2] >= cfg.negative_concern][:2]

    parts: list[str] = []
    if top_pos:
        if len(top_pos) == 1:
            parts.append(f"Kullanıcılar ürünün özellikle {top_pos[0]} yönünü genel olarak beğeniyor.")
        elif len(top_pos) == 2:
            parts.append(f"Kullanıcılar ürünün {top_pos[0]} ve {top_pos[1]} yönlerini genel olarak beğeniyor.")
        else:
            parts.append(
                f"Kullanıcılar ürünün {top_pos[0]}, {top_pos[1]} ve {top_pos[2]} yönlerini genel olarak beğeniyor."
            )
    else:
        parts.append("Kullanıcı yorumlarında belirgin bir güçlü yön tek başına öne çıkmıyor.")

    if top_neg:
        if len(top_neg) == 1:
            parts.append(f"En çok şikayet edilen konu ise {top_neg[0]} olarak görünüyor.")
        else:
            parts.append(f"En çok şikayet edilen konular ise {top_neg[0]} ve {top_neg[1]}.")
    else:
        parts.append("Yorumlarda belirgin bir olumsuzluk öne çıkmıyor.")

    return " ".join(parts)


def transform(data: dict[str, Any], cfg: Config) -> dict[str, Any]:
    overall = data.get("overall_sentiment", {}) or {}
    aspects = data.get("aspects", []) or []
    aspects = filter_aspects(aspects, cfg.IGNORE_ASPECTS)

    transformed_aspects: list[dict[str, Any]] = []
    for a in aspects:
        name = str(a.get("aspect", "")).strip()
        dist = a.get("sentiment", {}) or {}
        dist_norm = {
            "positive": float(dist.get("positive", 0.0)),
            "negative": float(dist.get("negative", 0.0)),
            "neutral": float(dist.get("neutral", 0.0)),
        }
        transformed_aspects.append(
            {
                "aspect": name,
                "duygu": get_sentiment_label(dist_norm, cfg),
                "ozet": build_aspect_summary(name, dist_norm, cfg),
            }
        )

    output = {
        "urun_geneli": {
            "genel_duygu": {
                "positive": format_pct(float(overall.get("positive", 0.0))),
                "negative": format_pct(float(overall.get("negative", 0.0))),
                "neutral": format_pct(float(overall.get("neutral", 0.0))),
            },
            "genel_yorum": build_general_comment(aspects, cfg),
        },
        "belirlenen_aspectler": transformed_aspects,
    }
    return output


def resolve_default_input(path: Path) -> Path:
    if path.exists():
        return path
    alt = Path("final_sentiment_output.json")
    if alt.exists():
        return alt
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Teknik sentiment JSON ciktisini dogal dil final rapora donustur")
    parser.add_argument("--input", type=Path, default=Config.input_json, help="Girdi JSON yolu")
    parser.add_argument("--output", type=Path, default=Config.output_json, help="Cikti JSON yolu")
    args = parser.parse_args()

    cfg = Config(input_json=resolve_default_input(args.input), output_json=args.output)
    raw = load_json(cfg.input_json)
    final_output = transform(raw, cfg)
    cfg.output_json.write_text(json.dumps(final_output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Girdi  : {cfg.input_json}")
    print(f"Cikti  : {cfg.output_json}")
    print("\nGenel duygu:")
    print(json.dumps(final_output["urun_geneli"]["genel_duygu"], ensure_ascii=False, indent=2))
    print("\nGenel yorum:")
    print(final_output["urun_geneli"]["genel_yorum"])


if __name__ == "__main__":
    main()

