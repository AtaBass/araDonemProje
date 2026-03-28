"""
Pipeline: 10 Kategori x 30 Ürün x 50 Yorum = 15.000 Yorum

Çalıştırma:
    python3 pipeline.py                  # tüm kategorileri çalıştır
    python3 pipeline.py --only sandalye  # sadece bir kategori
    python3 pipeline.py --resume         # kaldığı yerden devam et

Çıktı:
    output/{kategori}.csv                kategori başına tek dosya
    output/urls_{kategori}.json          keşfedilen URL'ler (önbellek)
    output/pipeline_state.json           ilerleme durumu
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from discover_products import discover_trendyol
from trendyol_scraper import scrape_trendyol

# ──────────────────────────────────────────────
# KATEGORİ TANIMI
# ──────────────────────────────────────────────
CATEGORIES = {
    "sandalye":          "sandalye",
    "spor_ayakkabi":     "spor ayakkabı",
    "kulaklık":          "kablosuz kulaklık",
    "airfryer":          "airfryer",
    "erkek_kot":         "erkek kot pantolon",
    "nevresim":          "nevresim takımı",
    "mouse":             "mouse",
    "akilli_saat":       "akıllı saat",
    "kadin_canta":       "kadın çanta",
    "kadin_tayt":        "kadın tayt",
}

# ──────────────────────────────────────────────
# AYARLAR
# ──────────────────────────────────────────────
MAX_PRODUCTS      = 30    # kategori başına ürün
MAX_REVIEWS       = 50    # ürün başına yorum
REVIEW_MAX_PAGES  = 10    # yorum sayfası üst sınırı (50 yorum için genellikle 5-6 yeter)
DELAY             = 1.2   # sayfa istekleri arası bekleme (saniye)
DISCOVERY_PAGES   = 15    # ürün keşfi için taranacak sayfa sayısı

OUTPUT_DIR        = Path("output")
STATE_FILE        = OUTPUT_DIR / "pipeline_state.json"


# ──────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ──────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def urls_cache_path(category: str) -> Path:
    return OUTPUT_DIR / f"urls_{category}.json"


def load_urls(category: str):
    path = urls_cache_path(category)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_urls(category: str, urls: list[str]):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(urls_cache_path(category), "w", encoding="utf-8") as f:
        json.dump(urls, f, ensure_ascii=False, indent=2)


def category_csv_path(category: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / f"{category}.csv"


def extract_content_id(url: str) -> str:
    import re
    m = re.search(r"-p-(\d+)", url)
    return m.group(1) if m else url.split("/")[-1][:20]


def print_banner(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def eta_str(done: int, total: int, elapsed: float) -> str:
    if done == 0:
        return "hesaplanıyor..."
    per_item = elapsed / done
    remaining = per_item * (total - done)
    h = int(remaining // 3600)
    m = int((remaining % 3600) // 60)
    return f"~{h}s {m}dk kaldı"


# ──────────────────────────────────────────────
# KEŞIF AŞAMASI
# ──────────────────────────────────────────────

def discover_category(category: str, query: str, force: bool = False) -> list[str]:
    cached = load_urls(category)
    if cached and not force:
        print(f"  [Önbellek] {len(cached)} URL yüklendi: {urls_cache_path(category)}")
        return cached

    print(f"\n[Keşif] '{query}' için Trendyol taranıyor ({DISCOVERY_PAGES} sayfa, hedef {MAX_PRODUCTS} ürün)...")
    urls = discover_trendyol(
        query_or_url=query,
        max_products=MAX_PRODUCTS,
        max_pages=DISCOVERY_PAGES,
    )

    if not urls:
        print(f"  UYARI: '{category}' için ürün bulunamadı!")
        return []

    save_urls(category, urls)
    print(f"  {len(urls)} URL kaydedildi → {urls_cache_path(category)}")
    return urls


# ──────────────────────────────────────────────
# YORUM ÇEKME AŞAMASI
# ──────────────────────────────────────────────

def scrape_category(category: str, urls: list[str], state: dict) -> dict:
    """Kategorideki tüm ürünlerin yorumlarını çeker. state güncellenmiş olarak döner."""

    cat_state = state.setdefault(category, {})
    total = len(urls)
    done_before = sum(1 for s in cat_state.values() if s == "done")

    print_banner(f"KATEGORİ: {category.upper()} ({done_before}/{total} tamamlandı)")

    start = time.time()
    done_this_run = 0
    category_rows = []

    for i, url in enumerate(urls):
        content_id = extract_content_id(url)
        status = cat_state.get(content_id, "pending")

        if status == "done":
            print(f"  [{i+1}/{total}] Atlanıyor (zaten tamamlandı): {content_id}")
            continue

        print(f"\n  [{i+1}/{total}] {url}")
        try:
            df = scrape_trendyol(
                url=url,
                max_pages=REVIEW_MAX_PAGES,
                delay=DELAY,
                output_csv=None,
                max_reviews=MAX_REVIEWS,
            )

            if df.empty:
                print(f"  UYARI: Yorum çekilemedi — {url}")
                cat_state[content_id] = "empty"
            else:
                # İstenen sütunları kategori CSV'si için normalize et
                selected = pd.DataFrame(
                    {
                        "urun adi": df.get("urun_adi", ""),
                        "urun url": df.get("urun_url", url),
                        "urun yorumu": df.get("urun_yorumu", df.get("yorum", "")),
                        "puan": df.get("puan", 0),
                        "tarih": df.get("tarih", ""),
                    }
                )
                category_rows.append(selected)
                cat_state[content_id] = "done"
                done_this_run += 1
                total_done = sum(1 for s in cat_state.values() if s == "done")
                elapsed = time.time() - start
                print(
                    f"  OK: {len(df)} yorum | "
                    f"Kategori: {total_done}/{total} | "
                    f"ETA: {eta_str(done_this_run, total - done_before, elapsed)}"
                )

        except Exception as e:
            print(f"  HATA: {e}")
            cat_state[content_id] = "error"

        # Her üründen sonra state kaydet
        save_state(state)

    # Kategori bitince tek CSV'ye yaz
    out_csv = category_csv_path(category)
    if category_rows:
        final_df = pd.concat(category_rows, ignore_index=True)
        final_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\n  Kategori CSV yazıldı: {out_csv} ({len(final_df)} satır)")
    else:
        # Başarılı ürün yoksa boş şema ile dosya bırak
        empty_df = pd.DataFrame(columns=["urun adi", "urun url", "urun yorumu", "puan", "tarih"])
        empty_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\n  Kategori CSV boş oluşturuldu: {out_csv}")

    return state


# ──────────────────────────────────────────────
# ANA PIPELINE
# ──────────────────────────────────────────────

def run_pipeline(only=None, force_discover: bool = False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()

    categories = (
        {only: CATEGORIES[only]}
        if only
        else CATEGORIES
    )

    if only and only not in CATEGORIES:
        print(f"Hata: '{only}' tanımlı kategoriler arasında değil.")
        print(f"Mevcut kategoriler: {', '.join(CATEGORIES)}")
        sys.exit(1)

    pipeline_start = time.time()
    total_reviews = 0

    for category, query in categories.items():
        # 1. Ürün URL'lerini keşfet
        urls = discover_category(category, query, force=force_discover)
        if not urls:
            continue

        # 2. Yorumları çek
        state = scrape_category(category, urls, state)

        # Özet
        cat_state = state.get(category, {})
        done = sum(1 for s in cat_state.values() if s == "done")
        print(f"\n  Kategori tamamlandı: {done}/{len(urls)} ürün başarılı")

        # Toplam yorum sayısını hesapla
        cat_csv = category_csv_path(category)
        if cat_csv.exists():
            try:
                df = pd.read_csv(cat_csv)
                total_reviews += len(df)
            except Exception:
                pass

    elapsed = time.time() - pipeline_start
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)

    print_banner(f"PIPELINE TAMAMLANDI")
    print(f"  Toplam süre   : {h}s {m}dk")
    print(f"  Toplam yorum  : {total_reviews:,}")
    print(f"  Çıktı klasörü : {OUTPUT_DIR.resolve()}")
    print(f"  State dosyası : {STATE_FILE.resolve()}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Trendyol yorum pipeline — 10 kategori × 30 ürün × 50 yorum",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python3 pipeline.py                          # tüm kategoriler
  python3 pipeline.py --only sandalye          # sadece sandalye
  python3 pipeline.py --resume                 # kaldığı yerden devam
  python3 pipeline.py --force-discover         # URL önbelleğini yenile
        """,
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help=f"Sadece bu kategoriyi çalıştır. Seçenekler: {', '.join(CATEGORIES)}"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Kaldığı yerden devam et (state dosyasını kullan)"
    )
    parser.add_argument(
        "--force-discover", action="store_true",
        help="URL önbelleğini yoksay, yeniden keşfet"
    )
    parser.add_argument(
        "--list-categories", action="store_true",
        help="Tanımlı kategorileri listele"
    )
    args = parser.parse_args()

    if args.list_categories:
        print("Tanımlı kategoriler:")
        for k, v in CATEGORIES.items():
            print(f"  {k:<20} → '{v}'")
        return

    if not args.resume:
        # Yeni çalıştırmada state sıfırlama isteğe bağlı
        existing_state = load_state()
        if existing_state and not args.only:
            done_total = sum(
                1
                for cat in existing_state.values()
                for s in cat.values()
                if s == "done"
            )
            print(f"Mevcut state bulundu: {done_total} ürün tamamlanmış.")
            print("Kaldığı yerden devam etmek için --resume, sıfırlamak için state dosyasını sil.")
            print(f"State: {STATE_FILE}")

    run_pipeline(only=args.only, force_discover=args.force_discover)


if __name__ == "__main__":
    main()
