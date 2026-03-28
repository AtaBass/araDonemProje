"""
Output klasöründeki CSV dosyalarını okuyup \"urun yorumu\" alanına hafif preprocessing uygular.
Orijinal dosyalara dokunulmaz; sonuçlar preprocess/ altına *_preprocessed.csv olarak yazılır.

Çalıştırma:
    python preprocessing.py
    python preprocessing.py --input-dir ../output --output-dir ../preprocess
    python preprocessing.py --min-chars 12
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

# Girdi / çıktı sütunları
COL_REVIEW_IN = "urun yorumu"
COL_PRODUCT = "urun adi"
COL_URL = "urun url"
COL_SCORE = "puan"
COL_DATE = "tarih"
COL_REVIEW_OUT = "temiz urun yorum"

OUTPUT_COLUMNS = [
    COL_PRODUCT,
    COL_URL,
    COL_REVIEW_OUT,
    COL_SCORE,
    COL_DATE,
]

# Türkçe İ/I için güvenli küçük harf (Unicode varsayılanı yeterli değil)
_TURKISH_CASE_MAP = str.maketrans("İIĞÜŞÖÇ", "iığüşöç")

# Emoji ve ilgili semboller (regex; ek bağımlılık yok)
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0000200D"
    "\U0000FE0F"
    "\U0001F3FB-\U0001F3FF"  # tenk tonu
    "]+",
    flags=re.UNICODE,
)

# Çok kısa yorum eşiği (karakter; boşluk sonrası)
DEFAULT_MIN_CHARS = 10


def _resolve_default_input_dir() -> Path:
    """Önce cwd/output, yoksa repo üst dizinindeki output (AraDProje/output)."""
    cwd_out = Path.cwd() / "output"
    if cwd_out.is_dir():
        return cwd_out.resolve()
    script_parent = Path(__file__).resolve().parent
    parent_out = script_parent.parent / "output"
    if parent_out.is_dir():
        return parent_out.resolve()
    return cwd_out.resolve()


def _resolve_default_output_dir(input_dir: Path) -> Path:
    """preprocess/ genelde output ile aynı üst dizinde."""
    return (input_dir.parent / "preprocess").resolve()


def turkish_lower(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    s = text.translate(_TURKISH_CASE_MAP)
    return s.lower()


def remove_emoji(text: str) -> str:
    if not text:
        return ""
    return _EMOJI_PATTERN.sub(" ", text)


def _is_word_char(ch: str) -> bool:
    if not ch:
        return False
    if ch.isalnum():
        return True
    # Türkçe harfler isalpha ile genelde gelir; ek güvence
    return ch.isalpha()


def simplify_punctuation(text: str) -> str:
    """
    Noktalama sadeleştirme (hafif):
    - Unicode tırnak ve tire varyantlarını sadeleştir
    - Ardışık noktalama işaretlerini tek boşluğa indir
    - Po (punctuation) kategorisindeki karakterleri boşlukla değiştir;
      ondalık nokta (3.5), kelime-içi tire ve apostrof korunur.
    """
    if not text:
        return ""

    # Tırnak / tire normalizasyonu
    for a, b in (
        ("\u201c", '"'),
        ("\u201d", '"'),
        ("\u2018", "'"),
        ("\u2019", "'"),
        ("\u2013", "-"),
        ("\u2014", "-"),
        ("\u2026", " "),
    ):
        text = text.replace(a, b)

    # Ardışık aynı noktalama
    text = re.sub(r"\.{3,}", " ", text)
    text = re.sub(r"([!?])\1{1,}", r"\1 ", text)

    chars = list(text)
    out: list[str] = []
    for i, c in enumerate(chars):
        cat = unicodedata.category(c)
        if not cat.startswith("P"):
            out.append(c)
            continue

        prev_ch = out[-1] if out else ""
        next_ch = chars[i + 1] if i + 1 < len(chars) else ""

        # Ondalık ayırıcı: rakam.rakam
        if c == "." and prev_ch.isdigit() and next_ch.isdigit():
            out.append(c)
            continue

        # Kelime-içi tire / apostrof
        if c in "-'" and _is_word_char(prev_ch) and _is_word_char(next_ch):
            out.append(c)
            continue

        out.append(" ")
    return "".join(out)


def collapse_whitespace(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def preprocess_comment(text: str) -> str:
    """Tek bir yorum metni üzerinde hafif preprocessing."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text).strip()
    if not s:
        return ""
    s = remove_emoji(s)
    s = simplify_punctuation(s)
    s = turkish_lower(s)
    s = collapse_whitespace(s)
    return s


def output_filename_for(input_path: Path) -> str:
    """sandalye.csv -> sandalye_preprocessed.csv"""
    return f"{input_path.stem}_preprocessed{input_path.suffix}"

def output_text_filename_for(input_path: Path) -> str:
    """sandalye.csv -> sandalye_preprocessed.txt (tab ayırıcı)"""
    return f"{input_path.stem}_preprocessed.txt"


def read_input_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def process_dataframe(df: pd.DataFrame, min_chars: int) -> pd.DataFrame:
    """urun yorumu -> temiz urun yorum; kısa satırları ele."""
    # Olası eski sütun adı
    review_col = COL_REVIEW_IN
    if review_col not in df.columns and "urun_yorumu" in df.columns:
        review_col = "urun_yorumu"

    required = [COL_PRODUCT, COL_URL, COL_SCORE, COL_DATE]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik sütunlar: {missing}. Mevcut: {list(df.columns)}")

    if review_col not in df.columns:
        raise ValueError(f"'{COL_REVIEW_IN}' sütunu yok. Mevcut: {list(df.columns)}")

    work = df.copy()
    work[COL_REVIEW_OUT] = work[review_col].map(preprocess_comment)
    # Çok kısa veya boş yorumları çıkar
    mask = work[COL_REVIEW_OUT].str.len() >= min_chars
    work = work.loc[mask].copy()

    out = work[OUTPUT_COLUMNS].copy()
    # Excel uyumu: düzgün tablo
    for col in OUTPUT_COLUMNS:
        out[col] = out[col].astype(str)
    return out


def process_file(
    input_path: Path,
    output_dir: Path,
    min_chars: int,
) -> tuple[Path, int]:
    df = read_input_csv(input_path)
    out_df = process_dataframe(df, min_chars=min_chars)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / output_filename_for(input_path)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig", lineterminator="\n")

    # Excel'de tablo şeklinde daha sorunsuz açılsın diye ayrıca tab-ayırıcı txt
    out_txt_path = output_dir / output_text_filename_for(input_path)
    out_df.to_csv(
        out_txt_path,
        index=False,
        encoding="utf-8-sig",
        sep="\t",
        lineterminator="\n",
    )

    return out_path, len(out_df)


def run_batch(
    input_dir: Path,
    output_dir: Path,
    min_chars: int,
) -> list[Path]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Girdi klasörü bulunamadı: {input_dir}")

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"Uyarı: {input_dir} içinde CSV yok.")
        return []

    written: list[Path] = []
    for path in csv_files:
        # Zaten preprocess çıktısı olan dosyaları atla (yanlışlıkla tekrar işleme)
        if path.stem.endswith("_preprocessed"):
            continue
        try:
            out, n_rows = process_file(path, output_dir, min_chars=min_chars)
            written.append(out)
            print(f"OK: {path.name} -> {out.name} ({n_rows} satır)")
        except Exception as e:
            print(f"HATA: {path.name}: {e}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Output CSV'lerinde urun yorumu preprocessing; sonuç preprocess/ altında.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Kaynak CSV klasörü (varsayılan: cwd/output veya ../output)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="İşlenmiş CSV klasörü (varsayılan: input'un üstünde preprocess/)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help=f"Bu uzunluğun altındaki temiz yorumlar atılır (varsayılan: {DEFAULT_MIN_CHARS})",
    )
    args = parser.parse_args()

    input_dir = args.input_dir or _resolve_default_input_dir()
    output_dir = args.output_dir or _resolve_default_output_dir(input_dir)

    print(f"Girdi : {input_dir}")
    print(f"Çıktı : {output_dir}")
    print(f"Min karakter: {args.min_chars}")
    run_batch(input_dir, output_dir, min_chars=args.min_chars)
    print("Bitti.")


if __name__ == "__main__":
    main()
