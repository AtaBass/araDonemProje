"""
Ürün Keşif Scraper - Playwright ile Trendyol & Hepsiburada
Arama kelimesi veya kategori URL'si verince ürün URL listesi döndürür.
"""

import re
import time
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout


def _trendyol_search_url(query: str, page: int = 1) -> str:
    from urllib.parse import quote
    return f"https://www.trendyol.com/sr?q={quote(query)}&pi={page}"


def _hepsiburada_search_url(query: str, page: int = 1) -> str:
    from urllib.parse import quote
    return f"https://www.hepsiburada.com/ara?q={quote(query)}&sayfa={page}"


def discover_trendyol(query_or_url: str, max_products: int = 20, max_pages: int = 3) -> list[str]:
    """
    Trendyol'da arama yap ve ürün URL'lerini döndür.

    Args:
        query_or_url: Arama kelimesi (örn: "laptop") veya kategori/arama URL'si
        max_products: Maksimum ürün sayısı
        max_pages: Taranacak maksimum sayfa sayısı

    Returns:
        Ürün URL'lerinin listesi
    """
    is_url = query_or_url.startswith("http")
    urls = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            locale="tr-TR",
            viewport={"width": 1280, "height": 800},
        )
        page = ctx.new_page()
        # Bot tespitini azalt
        page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        for p_num in range(1, max_pages + 1):
            if len(urls) >= max_products:
                break

            if is_url:
                sep = "&" if "?" in query_or_url else "?"
                target = query_or_url if p_num == 1 else f"{query_or_url}{sep}pi={p_num}"
            else:
                target = _trendyol_search_url(query_or_url, p_num)

            print(f"[Trendyol Keşif] Sayfa {p_num}: {target}")
            try:
                page.goto(target, wait_until="domcontentloaded", timeout=20000)
                page.wait_for_timeout(2000)
            except Exception as e:
                print(f"  Hata ({type(e).__name__}), devam ediliyor...")

            # Ürün linklerini topla
            links = page.eval_on_selector_all(
                "a[href*='-p-']",
                "els => els.map(e => e.href)"
            )
            found = 0
            for link in links:
                if re.search(r"-p-\d+", link):
                    # Sadece ürün sayfası, kategori/liste değil
                    clean = link.split("?")[0]
                    if clean not in urls:
                        urls.add(clean)
                        found += 1
                if len(urls) >= max_products:
                    break

            print(f"  {found} yeni ürün bulundu. Toplam: {len(urls)}")

            if found == 0:
                print("  Yeni ürün yok, durduruldu.")
                break

        browser.close()

    result = list(urls)[:max_products]
    print(f"[Trendyol Keşif] Toplam {len(result)} ürün URL'si bulundu.")
    return result


def discover_hepsiburada(query_or_url: str, max_products: int = 20, max_pages: int = 3) -> list[str]:
    """
    Hepsiburada'da arama yap ve ürün URL'lerini döndür.

    Args:
        query_or_url: Arama kelimesi (örn: "laptop") veya kategori/arama URL'si
        max_products: Maksimum ürün sayısı
        max_pages: Taranacak maksimum sayfa sayısı

    Returns:
        Ürün URL'lerinin listesi
    """
    is_url = query_or_url.startswith("http")
    urls = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            locale="tr-TR",
            viewport={"width": 1280, "height": 800},
        )
        page = ctx.new_page()
        page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        for p_num in range(1, max_pages + 1):
            if len(urls) >= max_products:
                break

            if is_url:
                sep = "&" if "?" in query_or_url else "?"
                target = query_or_url if p_num == 1 else f"{query_or_url}{sep}sayfa={p_num}"
            else:
                target = _hepsiburada_search_url(query_or_url, p_num)

            print(f"[Hepsiburada Keşif] Sayfa {p_num}: {target}")
            try:
                page.goto(target, wait_until="domcontentloaded", timeout=20000)
                page.wait_for_timeout(2000)
            except Exception as e:
                print(f"  Hata ({type(e).__name__}), devam ediliyor...")

            # Hepsiburada ürün URL'leri: /marka/urun-pm-HBVXXXXXX formatı
            links = page.eval_on_selector_all(
                "a[href*='-pm-'], a[href*='hepsiburada.com/']",
                "els => els.map(e => e.href)"
            )
            found = 0
            for link in links:
                # Gerçek ürün sayfası filtresi
                if re.search(r"-pm-[A-Z0-9]+", link) or re.search(r"hepsiburada\.com/[^/]+-[^/]+-p-\d+", link):
                    clean = link.split("?")[0]
                    if "hepsiburada.com" in clean and clean not in urls:
                        urls.add(clean)
                        found += 1
                if len(urls) >= max_products:
                    break

            print(f"  {found} yeni ürün bulundu. Toplam: {len(urls)}")

            if found == 0:
                print("  Yeni ürün yok, durduruldu.")
                break

        browser.close()

    result = list(urls)[:max_products]
    print(f"[Hepsiburada Keşif] Toplam {len(result)} ürün URL'si bulundu.")
    return result


def discover(site: str, query_or_url: str, max_products: int = 20, max_pages: int = 3) -> list[str]:
    """Siteye göre uygun keşif fonksiyonunu çağır."""
    if site == "trendyol":
        return discover_trendyol(query_or_url, max_products, max_pages)
    elif site == "hepsiburada":
        return discover_hepsiburada(query_or_url, max_products, max_pages)
    else:
        raise ValueError(f"Desteklenmeyen site: {site}")
