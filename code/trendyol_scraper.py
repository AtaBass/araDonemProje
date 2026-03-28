"""
Trendyol Yorum Scraper
Trendyol'un dahili review API'ını Playwright ile yakalar; CSS selector kullanmaz.
"""

import re
import time
from datetime import datetime, timezone

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

TR_MONTHS = {
    1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan",
    5: "Mayıs", 6: "Haziran", 7: "Temmuz", 8: "Ağustos",
    9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık",
}

STOP_WORDS = {
    "bir", "bu", "da", "de", "ve", "ile", "için", "ama", "çok", "en",
    "daha", "gibi", "kadar", "ki", "mi", "mu", "mü", "ne", "o", "ya",
    "benim", "beni", "ben", "şu", "biz", "siz", "hem", "hiç", "her",
    "bazı", "tüm", "böyle", "şöyle", "ise", "ya", "da",
}


def extract_content_id(url: str):
    m = re.search(r"-p-(\d+)", url)
    return m.group(1) if m else None


def ts_to_tr_date(ts_ms: int) -> str:
    """Unix milliseconds → '7 Mart 2026' formatı"""
    try:
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        return f"{dt.day} {TR_MONTHS[dt.month]} {dt.year}"
    except Exception:
        return ""


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(w for w in text.split() if w not in STOP_WORDS and len(w) > 1)


def _api_url(content_id: str, page: int, page_size: int = 20) -> str:
    return (
        "https://apigw.trendyol.com/discovery-storefront-trproductgw-service"
        f"/api/review-read/product-reviews/detailed"
        f"?contentId={content_id}&page={page}&pageSize={page_size}&channelId=1"
    )


def scrape_trendyol(
    url: str,
    max_pages: int = 10,
    delay: float = 1.0,
    output_csv: str = None,
    max_reviews: int = None,
) -> pd.DataFrame:
    """
    Trendyol ürün sayfasından yorum çeker.

    Strateji:
      1. İlk sayfayı tam yükle → API yanıtını intercept et (session cookie kurulur)
      2. Sonraki sayfalar için page.request ile API'ya doğrudan istek at
    """
    content_id = extract_content_id(url)
    base_url = url.split("?")[0]
    all_reviews = []
    product_name = ""
    total_pages = None  # API'dan öğrenilecek

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            locale="tr-TR",
            viewport={"width": 1440, "height": 900},
        )
        pg = ctx.new_page()
        pg.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

        # Her yorumSayfa=N için tarayıcıyı yönlendir, API yanıtını intercept et
        limit = max_pages if max_pages else 999

        for page_num in range(1, limit + 1):
            if max_reviews and len(all_reviews) >= max_reviews:
                break
            if total_pages is not None and page_num > total_pages:
                break

            target = f"{base_url}?yorumSayfa={page_num}"
            intercepted = {}

            def on_response(resp, _pn=page_num):
                if "product-reviews/detailed" in resp.url and "pageSize=20" in resp.url:
                    try:
                        intercepted["data"] = resp.json()
                    except Exception:
                        pass

            pg.on("response", on_response)

            print(f"    [Sayfa {page_num}] {len(all_reviews)} yorum toplandı...")
            try:
                pg.goto(target, wait_until="domcontentloaded", timeout=30000)
                pg.wait_for_timeout(1500)
                pg.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.6)")
                pg.wait_for_timeout(2500)
                if not product_name:
                    try:
                        product_name = pg.title().strip()
                        if " - Trendyol" in product_name:
                            product_name = product_name.replace(" - Trendyol", "").strip()
                    except Exception:
                        product_name = ""
            except Exception as e:
                print(f"    Hata ({type(e).__name__}, sayfa {page_num}), atlanıyor...")
                pg.remove_listener("response", on_response)
                time.sleep(delay)
                continue

            pg.remove_listener("response", on_response)

            if "data" not in intercepted:
                print(f"    Sayfa {page_num}: API yanıtı yok, duruldu.")
                break

            data = intercepted["data"]
            result = data.get("result", {})

            # Toplam sayfa sayısını ilk sayfadan öğren
            if total_pages is None:
                summary = result.get("summary", {})
                total_pages = summary.get("totalPages", 1)
                print(f"    Toplam yorum sayfası: {total_pages}")

            reviews_raw = result.get("reviews", [])
            if not reviews_raw:
                print(f"    Sayfa {page_num}: boş, duruldu.")
                break

            before = len(all_reviews)
            _parse_reviews(
                reviews_raw=reviews_raw,
                all_reviews=all_reviews,
                base_url=base_url,
                content_id=content_id,
                max_reviews=max_reviews,
                product_name=product_name,
            )
            print(f"    +{len(all_reviews) - before} yorum (toplam: {len(all_reviews)})")

            time.sleep(delay)

        browser.close()

    df = pd.DataFrame(all_reviews)

    if max_reviews:
        df = df.head(max_reviews)

    if output_csv and not df.empty:
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"    Kaydedildi: {output_csv} ({len(df)} yorum)")

    return df


def _parse_reviews(reviews_raw, all_reviews, base_url, content_id, max_reviews, product_name):
    for r in reviews_raw:
        if max_reviews and len(all_reviews) >= max_reviews:
            break

        yorum = r.get("comment", "") or ""
        ts = r.get("createdAt", 0)

        # Varyant bilgisi (attributes varsa)
        urun_renk, urun_beden = "", ""
        for attr in r.get("userProductAttributes", []) or []:
            name = (attr.get("name") or "").lower()
            val = attr.get("value") or ""
            if "renk" in name or "color" in name:
                urun_renk = val
            elif "beden" in name or "size" in name or "numara" in name:
                urun_beden = val

        all_reviews.append(
            {
                "urun_adi": product_name,
                "urun_url": base_url,
                "site": "Trendyol",
                "content_id": str(content_id or ""),
                "tarih": ts_to_tr_date(ts) if ts else "",
                "puan": r.get("rate", 0),
                "baslik": r.get("title", "") or "",
                "yorum": yorum,
                "urun_yorumu": yorum,
                "kullanici": r.get("userFullName", "") or "",
                "faydali": r.get("likesCount", 0) or 0,
                "faydasiz": r.get("dislikesCount", 0) or 0,
                "urun_renk": urun_renk,
                "urun_beden": urun_beden,
                "yorum_temiz": clean_text(yorum),
            }
        )
