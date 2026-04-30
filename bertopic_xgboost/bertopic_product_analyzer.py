"""
bertopic_product_analyzer.py
────────────────────────────
BERTopic tabanlı ürün-bazlı aspect analizi.

Yaklaşım:
  1. Tüm kategori verisi üzerinde BERTopic modeli eğitilir
     → semantik olarak anlamlı topic'ler öğrenilir (çok daha büyük corpus)
  2. Her ürünün yorumları bu modele transform edilir
  3. Aynı aspect adına düşen topic'ler BİRLEŞTİRİLİR (merge)
  4. Her aspect, o üründeki yorumların puanıyla skorlanır
  5. JSON çıktısı web sitesi için üretilir

Önce categorin tüm 1000+ yorumu üzerinde BERTopic çalışır →
anlamlı semantik topic'ler öğrenir → her topic'in c-TF-IDF 
kelimeleri bir aspect adına map edilir → tekrar edenler merge edilir.

Kullanım:
    python3 bertopic_product_analyzer.py --all
    python3 bertopic_product_analyzer.py --category mouse
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from hdbscan import HDBSCAN
from umap import UMAP
from xgboost import XGBClassifier

# ─── YOLLAR ────────────────────────────────────────────────────────
_THIS_DIR     = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
PREPROCESS_DIR = _PROJECT_ROOT / "preprocess"
WEB_DATA_DIR   = _THIS_DIR / "web" / "data" / "products"
MODEL_CACHE    = _THIS_DIR / ".model_cache"
MODEL_CACHE.mkdir(exist_ok=True)

CATEGORIES = [
    "airfryer", "akilli_saat", "erkek_kot", "kadin_canta",
    "kadin_tayt", "kulaklık", "mouse", "nevresim", "sandalye", "spor_ayakkabi",
]
CATEGORY_LABELS = {
    "airfryer":     "Airfryer",
    "akilli_saat":  "Akıllı Saat",
    "erkek_kot":    "Erkek Kot Pantolon",
    "kadin_canta":  "Kadın Çanta",
    "kadin_tayt":   "Kadın Tayt",
    "kulaklık":     "Kablosuz Kulaklık",
    "mouse":        "Mouse",
    "nevresim":     "Nevresim Takımı",
    "sandalye":     "Sandalye",
    "spor_ayakkabi":"Spor Ayakkabı",
}

# ─── TÜRKÇE METİN ─────────────────────────────────────────────────
_TR_MAP = str.maketrans("İIĞÜŞÖÇ", "iığüşöç")
def tr_lower(s: str) -> str:
    return str(s).translate(_TR_MAP).lower()

_EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001FAFF\U00002702-\U000027B0\U0000200D\U0000FE0F]+",
    re.UNICODE,
)
def clean(s: str) -> str:
    s = _EMOJI_RE.sub(" ", str(s))
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\d+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

# ─── STOP WORDS (Türkçe — genişletilmiş) ──────────────────────────
STOP_WORDS = {
    # Zamirler & edatlar
    "bir","bu","şu","o","da","de","için","ile","ve","ama","çok","en","az",
    "ben","sen","biz","siz","onlar","mi","mu","mı","mü","var","yok","ki",
    "ne","olan","oldu","olur","gibi","daha","hem","ya","ise","bile","kadar",
    "göre","çünkü","eğer","nasıl","neden","sadece","herkes","hiç","artık",
    # Fiiller (genel)
    "aldım","aldik","aldim","geldi","geldik","aldın","alındı","ulaştı",
    "beğendim","begendim","begendik","beğendik","bekliyorum","öneririm",
    "tavsiye","ederim","herkese","kesinlikle","gerçekten","gercekten",
    "sipariş","memnunum","memnun","memnunuz","teşekkür","tesekkur",
    "güzel","guzel","iyi","tam","harika","mükemmel","süper","çok",
    "falan","filan","sanki","yani","bilmiyorum","galiba","herhalde",
    # Genel ürün kelimeleri
    "ürün","urun","ürünü","ürünün","ürüne","paket","kargo","siparis",
    "hediye","arkadasim","arkadaşım","eşim","esim","annem","babam",
    "erkek","kadin","beden","numara","renk",
    # Zaman
    "gün","ay","yıl","saat","hafta","hızlı","hızla",
    # Eylemler
    "aldın","alındı","geldi","gitti","baktım","baktik",
    # Zamirler
    "bana","sana","bize","size","onlara","benden","senden",
    "oluyor","oluyor","oldu","olmadı","olmaz","olarak","olan",
}

# ─── ASPECT İSİM ÜRETME (c-TF-IDF → anlamlı başlık) ──────────────
# ─── YARDIMCILAR ───────────────────────────────────────────────────
def url_to_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]

def score_to_sentiment(s: float) -> str:
    return "positive" if s >= 4.5 else ("neutral" if s >= 3.0 else "negative")

def score_to_label(s: float) -> str:
    if s >= 4.5: return "Olumlu"
    if s >= 3.8: return "Genel Olumlu"
    if s >= 2.5: return "Karışık"
    return "Olumsuz"

def stars(s: float) -> str:
    f = int(s)
    h = "½" if s - f >= 0.4 else ""
    return "★" * f + h + "☆" * (5 - f - (1 if h else 0))

def get_icon(_: str) -> str:
    return "💬"

# ─── DİNAMİK ASPECT İSİMLENDİRME ─────────────────────────────────
# Hiç hardcoded sözlük yok — sadece c-TF-IDF ağırlıklarına göre
# en ayırt edici kelimeleri birleştirerek başlık üretilir.

# Sıklıkla geçip anlam taşımayan Türkçe eylem/zamir/edat kökleri
# (stop_words'ten farklı: kısa-anlamsız değil, topic-ayırt edici olmayan)
_NOISE: set[str] = {
    # Genel zarf / bağlaç
    "gibi","daha","çok","var","yok","hem","ise","bile","kadar",
    "zaten","artık","hala","sadece","yine","gene","tekrar","gayet",
    "oldukça","aslında","hakikaten","inanılmaz","kesinlikle","gerçekten",
    "gercekten","herkese","falan","filan","sanki","yani","galiba","herhalde",
    # Genel değerlendirme sıfatları (anlam taşımayan)
    "güzel","guzel","harika","mükemmel","süper","berbat","kötü",
    "iyiydi","güzeldi","kötüydü","berbattı","iyidir","güzeldir",
    "muhteşem","fevkalade","memnunum","memnunuz","memnun",
    # Teşekkür / selamlama
    "teşekkür","tesekkur","teşekkürler","sağolun","elinize","sağlık",
    "tavsiye","öneririm","alabilirsiniz","öneririm","ederim",
    # Fiil çekimleri (review dili)
    "oldu","olan","olarak","olmuş","oluyor","olmaz","olmadı","olduğu",
    "aldım","aldik","aldim","alındı","aldıktan","almıştık","aldıkları",
    "geldi","geldik","gitti","gelen","geliyor","götürdüm",
    "beğendim","begendim","begendik","beğendik","beğendi","begendi",
    "bekliyorum","bekliyordum","ettim","etmek","ettik",
    "kullandım","kullanıyorum","kullandık",
    "kaldım","kaldık","kaldı","verdi","verdim","verdik",
    "gördüm","gördük","düşündüm","sevdim","seviyorum",
    "geçti","geçen","çalışıyorum","çalışmadı","ediyorum","ediyoruz",
    # Kişi / aile / ilişki
    "oğlum","kızım","eşim","annem","babam","arkadaşım","arkadaşıma",
    "arkadasim","oğluma","kızıma","kardeşim","kardeşime","kardeşimin",
    "kendim","kendi","bana","sana","bize","size","onlara",
    "abim","abime","ablam","ablama","amcam","teyzem","dayım",
    # Sahiplik ekleri (bağlamdan kopuk)
    "içinde","içindeki","içine","üstünde","yanında","altında",
    # Zaman / zaman zarfları
    "gün","hafta","yıl","dakika","saniye","hemen","sonra","önce",
    "ilkinde","başta","henüz","daha","artık","şimdi",
    "geçmeden","geçmişte","bugün","dün","yarın",
    # Teslimat / ulaşım fiil ve zarfları
    "ulastı","ulaştı","ulaşmadı","ulaşmış","ulaştırdı",
    "gelince","gelir","gelmez","geldiğinde","geldikten",
    "günde","günü","günlük","gününde","günler","günlerdir",
    "hızlıca","çabuk","beklemeden","beklenmeden",
    # Yön / şekil / beden parçaları
    "şekilde","biçimde","tarzda","elime","elinde","elimde",
    "ayağım","ayağıma","ayağına","ayaklarım","bacağım",
    # Aile ekleri
    "anneme","babama","ablama","abime","kardeşime","oğluma","kızıma","eşime",
    # Sıralı sayılar / belirsiz nicelik
    "ikinci","üçüncü","dördüncü","birinci","ilkinde",
    "birkaç","bazı","çoğu","tümü","hepsi",
    # Kargo/teslimat ifadeleri (locative/adjective)
    "kargoda","kargoya","kargoyla","kargom","paketimde",
    "hızlıydı","hızlıdır","yavaştı","gecikmeli","zamanında",
    # Diğer gürültü
    "özenli","özenle","dikkatli","dikkatle",
    # Ürün URL kalıpları / teknik olmayan
    "mouse","mause","fare","ürünü","ürünün","ürüne",
    # Satıcı / sipariş (aspect değil)
    "satıcı","satıcıya","satıcının","sipariş","siparis",
    "hediye","hediyem","hediyemi","hediyeniz","hediyemiz",
    "fiyatına","ücrete","faturai",
    # Emir kipi / istek / dilek
    "aldırın","aldırınız","alınız","alın","alınız","alırsınız",
    "beğeneceksiniz","deneyin","denemeliyim","tercih","tercihim",
    "vermeye","verirsiniz","verildi","verilmeli","vermeyin",
    "inşallah","umarım","umarız","keşke","istiyorum","isterdim",
    # Bağlama / bağıntı kelimeleri
    "hemde","üstelik","sayesinde","sayesine","ayrıca","bunun",
    "bunlar","bunları","bunlarla","bunlardan","bunlara",
    "nedeniyle","sebebiyle","dolayı","itibaren","yorumlara",
    "yorumlar","yorumlarda","yorumlardaki",
    # Sahiplik zamirleri
    "benim","senin","bizim","sizin","onun","onların","benimki",
    "kendime","kendine","kendinize","kendisi","kendimle",
    "olsun","olması","olmalı","olacak","olacağı",
    # Belirsiz niteleyiciler
    "fazla","fazlası","biraz","epey","oldukca","neredeyse",
    "maalesef","neyse","yeterince","oldukça",
    "aşırı","inanılmaz","müthiş","fevkalade","mükemmel","süper",
    # Duygu / genel değerlendirme / beğeni
    "gönül","gönlüm","gönlünce","mutlu","mutluyum","mutluyuz",
    "memnun","memnunum","şikayetim","sorunum",
    "helal","maşallah","aferin","bravo","süper","harika",
    "iyiki","iyikim","doğru","doğrusu",
    # Zarf / bağlaç fazlası
    "hediyesi","hediyemdi","hediyelik",
    # Kişisel beden/ölçü ifadeleri
    "kiloyum","kilom","boyum","kilolarım","boylarım",
    "ederiz","ederek","etmiş","yaparız","yapıyoruz",
    # Zamanla ilgili
    "doğum","yıldönümü","bayram","özel","etkinlik",
    # Giyim fiil kökü
    "giymek","giymeli","giyiyor","giyiyorum","giyiyoruz",
    "ayakta","ayağında","ayaklarında",
    # Karar / düşünme ifadeleri
    "düşünmeden","tereddütsüz","çekinmeden","pişman","pişmanlık",
    "düşündüm","düşündük","düşünüyordum",
    # Genel sıfatlar / zarflar (anlam taşımayan)
    "başarılı","başarılıdır","başarısız","böyle","şöyle","öyle",
    "normalde","genelde","genellikle","çoğunlukla","bazen","nadiren",
    # Hal ekleri / konum
    "kutusunda","kutusundan","kutusuna","salonunda","salonundan",
    "içerisinde","içerisinden","dışında","yanında","üzerinde",
    # Fiil çekimi (geniş/gelecek)
    "alacağım","alacağız","alacaklar","alacaksın","alırsınız",
    "beğenildi","beğenilmiş","beğenilmedi","beğenildik",
    "sizden","bizden","senden","benden","ondan","onlardan",
    # Çıktı / etki
    "alalı","almaya","almayın","almadım","almıştım",
    # Beden / ölçü karşılaştırma (yorum kalıbı)
    "büyük","küçük","büyüğü","küçüğü","büyütmek","küçültmek",
    # Kişisel fikir / bağlaç (aspect değil)
    "hatta","bence","bana göre","diğer","birde","ayrıca",
    "zira","ancak","oysa","oyse","yoksa","belki","sanırım",
    # Olumsuzluk / genel terim
    "değil","değildi","değilmiş","değildir",
    # Yorum kalıpları — review cliché
    "tereddüt","etmeden","bedeninizi","vücudunuzu",
    "bedenimi","bedenime","bedeninize","bedenin","bedenim",
    "düşünmeden","çekinmeden","almanızı","almanızı tavsiye",
    # Fiil (geçmiş) — ek eksikli
    "oturdu","oturmuş","oturmak","otururken","oturunca",
    "durdu","durmuş","duruyor","duruyordu",
    # Belirsiz niteleyici devam
    "elinize","elinizde","kollarım","koluma","boynuma",
    # Türkçe konuşma kalıpları / argo intensifier
    "resmen","gerçekten","hakikaten","açıkçası","inanın",
    "inanılmaz","muhteşem","fevkalade","vallahi","yemin",
    # Neden / sonuç bağlaçları
    "yüzden","nedeniyle","dolayısıyla","bundan","ondan",
    # Kişisel konum ekleri
    "üstüme","üstünde","üstünü","üstümde","üstümdeki",
    "üzerime","üzerinde","üzerinden","yanımda","yanımıza",
    # Yanlış yazım / Türkçe olmayan
    "hizli","hızlı","hizlica","hizla",
    # Genel eylem tamamlayıcıları
    "aldıktan","geldikten","baktıktan","kullandıktan",
    # Belirsiz genel sıfatlar (ürün özelliği taşımayan)
    "tatlı","güzelce","ince","kalın","serin","sıcak","soğuk",
    # Ürün kategorisi kelimeleri (her yorumda geçer, ayırt edici değil)
    "ayakkabı","ayakkabi","kulaklık","nevresim","çanta",
    "sandalye","airfryer","fritöz","mouse","mause",
    "tayt","pantolon","sweatshirt","tişört","gömlek",
    # Paket/teslimat fiil çekimleri
    "paketlenmişti","paketlendi","paketlenmiş","gönderildi","gönderilmiş",
    "teslim","teslimde","teslimatta",
    # Numara/beden tavsiye kalıpları (kişisel 2. çoğul)
    "numaranızı","numaranızı alın","bedeninizi alın","numaranızı almak",
    "numaranızı","boyunuzu","boyunuzu alın",
    # Yazım yanlışları (ayakkabı varyantları)
    "ayakabı","ayakabi","ayakabıyı","ayakkabıyı",
    # Kişisel 2. çoğul ekli kelimeler (yorum tavsiyesi kalıbı)
    "hesabınızı","sayfanızı","sitenizi","alışverişinizi",
    # ── YENİ: Kalite/özellik sıfatları — aspect adına GİRMEMELİ ──────
    # Bunlar her yorumda geçen genel değerlendirme sıfatları;
    # aspect ADINDA değil, aspect İÇERİĞİNDE kullanılmalı.
    "kaliteli","kalitesiz","kaliteyi","kalitede","kaliteden","kaliteyle",
    "kalitesiyle","kalitesinin","kalitesinde","kalitesini",
    "dayanıklı","dayanıksız","dayanıklılık","dayanıklılığı",
    "sağlam","sağlamlık","sağlamlığı",
    "esnek","esneklik","esnekliği",
    "güvenilir","güvenilmez",
    "kullanışlı","kullanışsız","kullanışlılık",
    "rahat","rahatsız","rahatlık","rahatlığı",
    # Görsel/renk genel (aspect ayırt edici değil)
    "görseldeki","görselde","görsellerden","görsellerdeki","görseli",
    "renkli","renksiz","renginde","rengiyle",
    # Beden/kalıp sıfatları (çok genel)
    "bedende","bedenle","bedenin","bedenime",
    "numara","numarası","numarada",
    # Süper/harika çekimleri (yukarıda kökler var ama ekli formlar eksikti)
    "süperdi","süperdir","harikaydı","harikadır",
    # Fiyat çekimli formları (aspect'te "fiyat" kalmalı, çekimliler gürültü)
    "fiyata","fiyattan","fiyatı","fiyatla","fiyatında","fiyatından",
}

def _clean_word(w: str) -> str:
    w = tr_lower(w).strip()
    w = re.sub(r"[^\w]", "", w)
    return w

# ─── TÜRKÇE ÇEKİM EKİ SOYMA (aspect label normalizasyonu) ─────────
# Amaç: "kumaşı" → "kumaş", "kalıbı" → "kalıp" değil ama en azından
# "tasarımı" → "tasarım", "performansı" → "performans" gibi basit vakaları çöz.
_TR_LABEL_SUFFIXES: list[tuple[str, int]] = [
    # (sonek, minimum_kök_uzunluğu) — uzundan kısaya sıralı
    ("sından", 4), ("sinden", 4), ("sundan", 4), ("sünden", 4),
    ("sıyla",  4), ("siyle",  4), ("suyla",  4), ("süyle",  4),
    ("sında",  4), ("sinde",  4), ("sunda",  4), ("sünde",  4),
    ("sına",   4), ("sine",   4), ("suna",   4), ("süne",   4),
    ("sını",   4), ("sini",   4), ("sunu",   4), ("sünü",   4),
    ("ların",  4), ("lerin",  4),
    ("ları",   4), ("leri",   4),
    ("ından",  4), ("inden",  4), ("undan",  4), ("ünden",  4),
    ("ında",   4), ("inde",   4), ("unda",   4), ("ünde",   4),
    ("ına",    4), ("ine",    4), ("una",    4), ("üne",    4),
    ("ını",    4), ("ini",    4), ("unu",    4), ("ünü",    4),
    ("lar",    4), ("ler",    4),
    ("dan",    4), ("den",    4), ("tan",    4), ("ten",    4),
    ("ın",     4), ("in",     4), ("un",     4), ("ün",     4),
    ("sı",     4), ("si",     4), ("su",     4), ("sü",     4),
    ("yı",     4), ("yi",     4), ("yu",     4), ("yü",     4),
    # Tek karakter — yalnızca uzun kelimeler (6+) için güvenli
    ("ı",      5), ("i",      5), ("u",      5), ("ü",      5),
]

def _normalize_label_word(w: str) -> str:
    """Aspect label için Türkçe çekim ekini soy. Kök tahmin etmez,
    yalnızca soneki keser; harf dönüşümü (b→p, d→t) uygulanmaz."""
    for suf, min_root in _TR_LABEL_SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= min_root:
            return w[: -len(suf)]
    return w

def _is_meaningful(w: str) -> bool:
    """Ürün özelliği anlatan, aspect başlığına girebilecek kelime mi?"""
    if len(w) < 5:
        return False
    if any(c.isdigit() for c in w):
        return False
    if w in STOP_WORDS or w in _NOISE:
        return False
    # Fiil sonu ekleri → fiil, atla
    if re.search(
        r"(dım|dim|tım|tim|dik|dık|tık|tik"
        r"|arak|erek|iken|meli|malı"
        r"|acak|ecek"
        r"|ıyor|iyor|uyor|üyor"
        r"|yorum|yoruz|yorsun|yorlar"
        r"|ydım|ydim|ydık|ydik|ydı|ydi"
        r"|mek|mak"           # infinitif
        r"|dum|düm|tum|tüm"
        # Türkçe kişisel geçmiş/geniş zaman ekleri
        r"|mışım|mışız|mışsın|mişim|mişiz|mişsin"
        r"|muşum|muşuz|müşüm|müşüz"
        r"|acağım|acağız|acaksın|eceğim|eceğiz|eceksin"
        r"|ıyorum|iyorum|uyorum|üyorum"
        r"|ıyoruz|iyoruz|uyoruz|üyoruz"
        r"|ıyorsun|iyorsun"
        # Geçmiş zaman kişisel
        r"|dıktan|dikten|tıktan|tikten"
        r"|dığım|diğim|duğum|düğüm"
        r"|ınca|ince|unca|ünce"
        r"|arak|erek"
        # Bileşik geçmiş zaman (miş+ti, mış+tı vb.)
        r"|mişti|mıştı|muştu|müştü"
        r"|diydi|tiydi|tıydı|dıydı"
        # Şart kipi
        r"|seydi|saydı|seydik|saydık"
        r"|ırsam|irsem|ursam|ürsem"
        # Sürerli geçmiş
        r"|ıyordu|iyordu|uyordu|üyordu"
        r"|ıyorduk|iyorduk)$",
        w,
    ):
        return False
    # Kişisel hal ekleri (locative/dative kişisel ifadeler)
    if re.search(r"(ıma|ime|uma|üme|ama|eme|üme)$", w) and len(w) > 6:
        return False
    # Yön/kaynak hal ekleri kişisel
    if re.search(r"(ımdan|imden|umdan|ümden|amdan|emden)$", w):
        return False
    return True

def label_from_keywords(words: list[str], scores: list[float] | None = None) -> str | None:
    """
    c-TF-IDF kelimelerinden dinamik aspect başlığı üret.
    Öncelik: anlamlı bigram > tek güçlü kelime.
    & yalnızca iki kelimenin GÜÇ ORANI çok yakınsa ve anlamlıysa kullanılır.
    Anlamlı kelime bulunamazsa None döndür → topic atlanır.
    """
    total = len(words)

    def weight_of(rank: int, is_bigram: bool) -> float:
        base = scores[rank] if scores and rank < total else max(0.0, total - rank)
        return base * (1.5 if is_bigram else 1.0)

    def _display(w_clean: str) -> str:
        """Çekim ekini soy ve büyük harfle başlat."""
        normalized = _normalize_label_word(w_clean)
        # Normalize sonucu anlamsız kısaltmışsa orijinali kullan
        if len(normalized) < 4:
            normalized = w_clean
        return normalized.capitalize()

    # ── 1. Önce anlamlı bigram'lara bak ──
    best_bigram: tuple[float, str] | None = None
    for rank, w in enumerate(words):
        if " " not in w:
            continue
        parts = [_clean_word(p) for p in w.split()]
        # Duplikat bigram (örn. "kaliteli kaliteli") → atla
        if len(parts) == 2 and parts[0] == parts[1]:
            continue
        good = [p for p in parts if _is_meaningful(p)]
        if len(good) < 2:  # bigram'da HER İKİ kelime de anlamlı olmalı
            continue
        # Her iki kelime normalize edilmiş haliyle gösterilir
        display = " ".join(_display(p) for p in parts)
        wt = weight_of(rank, is_bigram=True)
        if best_bigram is None or wt > best_bigram[0]:
            best_bigram = (wt, display)

    # ── 2. Anlamlı unigram'ları topla ──
    unigram_candidates: list[tuple[float, str]] = []
    for rank, w in enumerate(words):
        if " " in w:
            continue
        w_clean = _clean_word(w)
        if not _is_meaningful(w_clean):
            continue
        unigram_candidates.append((weight_of(rank, is_bigram=False), _display(w_clean)))
    unigram_candidates.sort(key=lambda x: -x[0])

    # ── 3. Sonucu belirle ──
    if best_bigram:
        bg_w, bg_label = best_bigram
        if unigram_candidates:
            top_uni_w, top_uni = unigram_candidates[0]
            # Bigram yeterince güçlüyse tek başına döndür
            if bg_w >= top_uni_w * 0.75:
                return bg_label
            # Unigram daha güçlüyse ama bigram tamamlayıcıysa birleştir
            if top_uni.lower() not in bg_label.lower():
                return f"{top_uni} / {bg_label}"  # & yerine / — daha okunabilir
        return bg_label

    if not unigram_candidates:
        return None

    top1_w, top1 = unigram_candidates[0]
    if len(unigram_candidates) == 1:
        return top1

    top2_w, top2 = unigram_candidates[1]

    # İkinci kelime yalnızca birinciye çok yakın VE farklıysa birleştir
    # Eşik 0.6'dan 0.85'e yükseltildi → tek kelimelik temiz başlık tercih edilir
    if top2_w >= top1_w * 0.85 and top2.lower() not in top1.lower():
        return f"{top1} & {top2}"

    return top1

# ─── BERTopic MODELİ EĞİTİMİ ──────────────────────────────────────
def build_bertopic_model(texts: list[str], n_topics: int = 10) -> BERTopic:
    """
    Tüm kategori metinleri üzerinde BERTopic eğitir.
    n_topics: hedef topic sayısı (nr_topics parametre)
    """
    # Multilingual sentence transformer (Türkçe için iyi çalışır)
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # UMAP — küçük veri için n_neighbors küçük tutulur
    n = len(texts)
    n_neighbors = max(5, min(15, n // 10))
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=True,
    )

    # HDBSCAN — min_cluster_size veri boyutuna göre
    min_cluster = max(5, n // 40)
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster,
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # Vectorizer — gürültülü kelimeleri c-TF-IDF'den tamamen çıkar
    _all_stopwords = list(STOP_WORDS | _NOISE)
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=_all_stopwords,
        min_df=3,
        max_df=0.85,
        token_pattern=r"(?u)\b[a-züğışöçıİĞÜŞÖÇ]{4,}\b",
    )

    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        nr_topics=n_topics,          # hedef topic sayısı
        top_n_words=15,
        verbose=False,
        calculate_probabilities=True,
    )
    return model

# ─── ASPECT SONUCU ────────────────────────────────────────────────
@dataclass
class AspectResult:
    name: str
    review_count: int
    avg_score: float
    sentiment: str
    sentiment_label: str
    top_terms: list[str]
    positive_pct: float
    negative_pct: float
    neutral_pct: float
    sample_reviews: list[str]
    all_reviews: list[dict]
    icon: str = "💬"

def compute_aspect(
    df: pd.DataFrame,
    name: str,
    indices: list[int],
    terms: list[str],
) -> AspectResult:
    sub = df.iloc[indices]
    scores = sub["puan"].values
    avg = float(np.mean(scores))
    pos = float(np.mean(scores >= 4.5))
    neg = float(np.mean(scores <= 2.0))
    neu = max(0.0, 1.0 - pos - neg)
    # Tekrar eden yorumları temizle
    seen: set[str] = set()
    unique_rows = []
    for _, row in sub.sort_values("puan", ascending=False).iterrows():
        txt = row["temiz urun yorum"].strip()
        if txt and txt not in seen:
            seen.add(txt)
            unique_rows.append({"text": txt, "score": float(row["puan"])})

    all_reviews = unique_rows

    # Sample reviews: aspect top_terms'lerini içeren yorumları öncelikle göster
    term_set = {tr_lower(t) for t in terms}
    def relevance(txt: str) -> int:
        low = tr_lower(txt)
        return sum(1 for t in term_set if t in low)

    ranked = sorted(unique_rows, key=lambda r: (-relevance(r["text"]), -len(r["text"])))
    sample = [r["text"] for r in ranked[:3]]
    return AspectResult(
        name=name,
        review_count=len(indices),
        avg_score=round(avg, 2),
        sentiment=score_to_sentiment(avg),
        sentiment_label=score_to_label(avg),
        top_terms=terms,
        positive_pct=round(pos * 100, 1),
        negative_pct=round(neg * 100, 1),
        neutral_pct=round(neu * 100, 1),
        sample_reviews=sample,
        all_reviews=all_reviews,
        icon=get_icon(name),
    )

# ─── TÜRKÇE ÖZET (madde madde, aspect odaklı) ─────────────────────

def _best_phrase(review: str, terms: list[str], max_len: int = 85) -> str:
    """Yorumdan aspete en ilgili ve düzgün cümleyi çıkar."""
    sentences = re.split(r"[.!?\n]+", review)
    best, best_score = "", 0
    for s in sentences:
        s = s.strip()
        if len(s) < 15:
            continue
        score = sum(1 for t in terms if tr_lower(t) in tr_lower(s))
        if score > best_score or (score == best_score and len(s) > len(best)):
            best_score, best = score, s
    result = (best or review).strip()
    # Baştaki küçük harfi büyüt
    if result:
        result = result[0].upper() + result[1:]
    return result[:max_len].rstrip() + ("…" if len(result) > max_len else "")


def _dedupe_terms(terms: list[str]) -> list[str]:
    """Aynı kökten gelen tekrarlayan kelimeleri temizle (kumaş/kumaşı → tek).
    Aynı zamanda 'rahat rahat' gibi duplicate bigram'ları da filtreler."""
    seen_roots: set[str] = set()
    result = []
    for t in terms:
        # Duplicate bigram ("rahat rahat", "kaliteli kaliteli") → atla
        parts = tr_lower(t).split()
        if len(parts) == 2 and parts[0] == parts[1]:
            continue
        w = tr_lower(t)
        # Kök: kısa kelimelerde tamamı, uzunlarda ilk 5 harf
        root = w if len(w) <= 5 else w[:5]
        if root not in seen_roots:
            seen_roots.add(root)
            result.append(t)
    return result


# ─── EXTRACTİVE ÖZET — Template yok, yorumdan doğrudan çıkarım ────

# Genel övgü/eleştiri kalıpları — tek başına bilgi taşımayan cümleler atlanır
_GENERIC_PHRASES = {
    "çok güzel", "harika bir ürün", "mükemmel", "süper", "çok beğendim",
    "kesinlikle tavsiye", "herkese tavsiye", "alın derim", "pişman olmaz",
    "fiyatına göre", "beklentimi karşıladı", "memnun kaldım", "teşekkürler",
    "tam istediğim gibi", "çok işlevsel", "çok kullanışlı", "tam beklediğim",
}

def _sentence_score(sent: str, term_set: set[str]) -> float:
    """
    Bir cümlenin 'extractive özet' değerini ölç.
    Yüksek puan = aspect'e alaka + içerik zenginliği + özgünlük.
    """
    sent_lower = tr_lower(sent)
    words = sent_lower.split()

    # Çok kısa veya çok uzun cümleler elenir
    if len(words) < 4 or len(sent) > 130:
        return 0.0

    # Yalnızca genel övgü/eleştiri içeriyorsa 0
    if any(p in sent_lower for p in _GENERIC_PHRASES) and len(words) < 8:
        return 0.0

    # İçerik kelimesi sayısı (stop/noise değil, 4+ karakter)
    content_words = [
        w for w in words
        if len(w) >= 4 and w not in STOP_WORDS and w not in _NOISE
    ]
    if len(content_words) < 2:
        return 0.0

    # Keyword alaka puanı (ağırlık 3)
    kw_hits = sum(1 for t in term_set if t in sent_lower)

    # İçerik çeşitliliği (benzersiz içerik kelimesi / toplam)
    diversity = len(set(content_words)) / max(len(content_words), 1)

    # Cümle uzunluk bonusu (20-90 karakter ideal)
    length_bonus = 0.3 if 20 <= len(sent) <= 90 else 0.0

    return kw_hits * 3.0 + diversity + length_bonus


def _extractive_aspect_bullet(asp: AspectResult) -> str:
    """
    Aspect için review'lardan doğrudan en bilgilendirici cümleyi çıkar.
    Template KULLANMAZ — yorumların gerçek içeriğini yansıtır.

    Çıktı örneği:
      "**Kumaş**: Ürün kumaşı kaliteli ve likralı, esnek bir yapıya sahip."
      "**Kalıp**: Kalıplar dar olduğundan bir beden büyük alınması öneriliyor."
    """
    term_set = {tr_lower(t) for t in asp.top_terms[:8]}
    # Aspect adının kök kelimelerini de ekle (normalize)
    name_kws = {
        _normalize_label_word(tr_lower(w))
        for w in asp.name.replace("&", " ").replace("/", " ").split()
        if len(w) >= 4
    }
    term_set |= name_kws

    candidates: list[tuple[float, str]] = []
    seen_fp: set[str] = set()          # duplikat cümle filtresi

    # Tüm yorumlardan cümle havuzu oluştur (en fazla 25 yorum)
    for rev_dict in asp.all_reviews[:25]:
        review = rev_dict["text"]
        rev_score = rev_dict.get("score", 3.0)

        # Nokta, ünlem, soru işareti, virgül ve satır sonu ile böl
        for sent in re.split(r"[.!?\n]+", review):
            sent = sent.strip()
            if len(sent) < 18:
                continue

            score = _sentence_score(sent, term_set)
            if score <= 0:
                continue

            # Puan bonusu: olumlu review'dan gelen cümle biraz daha değerli
            if rev_score >= 4.5:
                score += 0.2
            elif rev_score <= 2.0:
                score += 0.15          # olumsuz aspect'lerde olumsuz yorumlar da değerli

            # Basit fingerprint ile duplikat engelle (ilk 18 karakter)
            fp = tr_lower(sent)[:18]
            if fp in seen_fp:
                continue
            seen_fp.add(fp)

            candidates.append((score, sent))

    if not candidates:
        # Hiç uygun cümle yoksa — en azından kısa bir durum ifadesi döndür
        sentiment_word = (
            "olumlu" if asp.avg_score >= 4.0
            else ("karışık" if asp.avg_score >= 3.0 else "olumsuz")
        )
        return f"**{asp.name}**: {asp.review_count} yorum incelendi; genel görüş {sentiment_word}."

    # En yüksek puanlı cümleyi seç
    candidates.sort(key=lambda x: -x[0])
    best = candidates[0][1].strip()

    # Temizlik: baş harfi büyüt, nokta ekle
    if best:
        best = best[0].upper() + best[1:]
    if best and not best[-1] in ".!?":
        best += "."

    return f"**{asp.name}**: {best}"


def generate_summary(
    product_name: str,
    aspects: list[AspectResult],
    overall: float,
    n_reviews: int,
) -> str:
    if not aspects:
        return f"{n_reviews} yorum analiz edilmiştir, belirgin bir konu tespit edilemedi."

    short = product_name.split(" - ")[0][:55]

    # Kısa, tek cümlelik giriş (template burada kabul edilebilir — genel bağlam verir)
    if overall >= 4.5:
        intro = f"**{short}** için {n_reviews} müşteri yorumu incelenmiştir. Genel değerlendirmeler oldukça olumludur."
    elif overall >= 4.0:
        intro = f"**{short}** için {n_reviews} yorum analiz edilmiştir. Genel memnuniyet yüksek, bazı noktalarda farklı görüşler mevcuttur."
    elif overall >= 3.0:
        intro = f"**{short}** için {n_reviews} yorum incelenmiştir. Görüşler karma bir tablo ortaya koymaktadır."
    else:
        intro = f"**{short}** için {n_reviews} yorum incelenmiştir. Olumsuz geri bildirimler ön plandadır."

    bullets: list[str] = []
    for asp in aspects[:5]:
        bullet = _extractive_aspect_bullet(asp)
        bullets.append("• " + bullet)

    return intro + "\n" + "\n".join(bullets)

# ─── KEYWORD TABANLI MULTI-LABEL ASPECT ATAMASI ───────────────────
# BERTopic → aspect adları + keyword setleri keşfeder.
# Her yorumun hangi aspect'e ait olduğu ise keyword eşleştirmeyle
# (multi-label) belirlenir: bir yorum birden fazla aspect'e dahil olabilir.

def _build_aspect_keyword_sets(
    topic_terms: dict[int, list[str]],
    topic_labels: dict[int, str],
) -> dict[str, list[str]]:
    """
    Her aspect için normalize edilmiş anahtar kelime listesi üret.
    BERTopic topic'lerinin c-TF-IDF kelimeleri, çekim ekleri soyularak
    ve kopyaları temizlenerek keyword setine dönüştürülür.
    """
    aspect_kws: dict[str, list[str]] = {}
    for tid, label in topic_labels.items():
        terms = topic_terms.get(tid, [])
        kws: list[str] = []
        seen: set[str] = set()
        # Yalnızca en ayırt edici 5 terimi kullan — daha geniş set, daha fazla yanlış eşleşme demek
        for t in terms[:5]:
            for part in t.split():
                w = tr_lower(_clean_word(part))
                if len(w) < 4 or w in STOP_WORDS or w in _NOISE:
                    continue
                if w not in seen:
                    seen.add(w)
                    kws.append(w)
                norm = _normalize_label_word(w)
                if len(norm) >= 4 and norm not in seen:
                    seen.add(norm)
                    kws.append(norm)
        if kws:
            aspect_kws[label] = kws
    return aspect_kws


def _assign_review_to_aspects(
    text_lower: str,
    aspect_kws: dict[str, list[str]],
    min_matches: int = 1,
) -> list[str]:
    """
    Bir yorumun hangi aspect'lerden bahsettiğini keyword eşleştirmeyle saptar.
    Multi-label: bir yorum birden fazla aspect döndürebilir.
    min_matches: kaç farklı keyword gerektiği (metin uzunluğuna göre dışardan belirlenir).
    """
    matched: list[str] = []
    for label, kws in aspect_kws.items():
        hit_kws = {kw for kw in kws if kw in text_lower}  # set → tekrarı önler
        if len(hit_kws) >= min_matches:
            matched.append(label)
    return matched


# ─── KATEGORİ ANALİZİ (BERTopic) ─────────────────────────────────
def analyze_category_bertopic(category: str, output_dir: Path, n_topics: int = 12) -> list[dict]:
    csv_path = PREPROCESS_DIR / f"{category}_preprocessed.csv"
    if not csv_path.exists():
        print(f"  ⚠ Dosya yok: {csv_path}")
        return []

    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df["puan"] = pd.to_numeric(df["puan"], errors="coerce").fillna(3.0)
    df["temiz urun yorum"] = df["temiz urun yorum"].fillna("").str.strip()
    df = df[df["temiz urun yorum"].str.len() >= 10].copy()
    df.reset_index(drop=True, inplace=True)

    print(f"  Toplam {len(df)} yorum ile BERTopic eğitiliyor...")

    # ── Tüm kategoride BERTopic eğit ──
    all_texts = [tr_lower(clean(t)) for t in df["temiz urun yorum"].tolist()]

    # Embeddings önceden üret — hem BERTopic hem XGBoost için kullanılır
    _st_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print(f"  Embedding üretiliyor ({len(all_texts)} metin)...")
    embeddings = _st_model.encode(all_texts, show_progress_bar=False, batch_size=64)

    model = build_bertopic_model(all_texts, n_topics=n_topics)

    try:
        topics, probs = model.fit_transform(all_texts, embeddings=embeddings)
    except Exception as e:
        print(f"  ✗ BERTopic hatası: {e}")
        return []

    df["topic"] = topics

    n_topics_found = model.get_topic_info().shape[0] - 1
    n_outliers = int((np.array(topics) == -1).sum())
    print(f"  {n_topics_found} topic keşfedildi. Outlier: {n_outliers}/{len(topics)} yorum.")

    # ── XGBoost: outlier yorumları topic'e ata ──────────────────────

    labeled_mask  = np.array(topics) != -1
    outlier_mask  = ~labeled_mask
    valid_topics  = [t for t in set(topics) if t != -1]

    if outlier_mask.sum() > 0 and labeled_mask.sum() >= 20 and len(valid_topics) >= 2:
        X_train = embeddings[labeled_mask]
        y_train = np.array(topics)[labeled_mask]

        le = LabelEncoder()
        y_enc = le.fit_transform(y_train)

        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )
        xgb.fit(X_train, y_enc)

        X_out = embeddings[outlier_mask]
        pred_enc = xgb.predict(X_out)
        pred_topics = le.inverse_transform(pred_enc)

        # Sadece mevcut geçerli topic'lere ata
        valid_set = set(valid_topics)
        out_indices = np.where(outlier_mask)[0]
        assigned = 0
        for i, pred in zip(out_indices, pred_topics):
            if pred in valid_set:
                df.at[i, "topic"] = int(pred)
                assigned += 1
        print(f"  XGBoost: {assigned}/{outlier_mask.sum()} outlier yorum topic'e atandı.")
    else:
        print(f"  XGBoost: yeterli eğitim verisi yok, outlier atlaması yapılmadı.")
    # ────────────────────────────────────────────────────────────────

    # ── Topic → Aspect adı (c-TF-IDF ile) ──
    topic_info = model.get_topic_info()
    # Ham topic → (label, clean_terms) haritası
    raw_topic_labels: dict[int, str] = {}
    topic_terms:      dict[int, list[str]] = {}

    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            continue
        word_score_pairs = model.get_topic(tid)
        words  = [w for w, _ in word_score_pairs]
        scores = [s for _, s in word_score_pairs]
        clean_pairs = [(w, s) for w, s in zip(words, scores)
                       if w not in STOP_WORDS and len(w) > 3]
        clean_words  = [w for w, _ in clean_pairs]
        clean_scores = [s for _, s in clean_pairs]
        label = label_from_keywords(clean_words, clean_scores)
        if label is None:
            continue  # anlamsız topic → atla
        raw_topic_labels[tid] = label
        topic_terms[tid] = clean_words[:10]

    # ── Aynı aspect adına düşen topic'leri MERGE et ──
    # label → [topic_id, ...] grup
    label_to_tids: dict[str, list[int]] = {}
    for tid, label in raw_topic_labels.items():
        label_to_tids.setdefault(label, []).append(tid)

    # merged_topic_groups[label] = tüm satır indeksleri (kategori genelinde)
    # Bu merged map, sonra her ürün için subset alınacak
    topic_labels = raw_topic_labels  # ürün bazında merge aşağıda yapılacak

    # ── Keyword tabanlı multi-label atama için ön hazırlık ──
    # Her aspect adından o aspect'in topic_id'sine ters harita
    label_to_tid: dict[str, int] = {label: tid for tid, label in topic_labels.items()}
    # Her aspect için keyword seti (BERTopic'in c-TF-IDF kelimelerinden)
    aspect_kws = _build_aspect_keyword_sets(topic_terms, topic_labels)
    print(f"  {len(aspect_kws)} aspect için keyword seti oluşturuldu.")

    # ── Her ürün için aspect analizi ──
    cat_dir = output_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    products_meta = df.groupby("urun url")["temiz urun yorum"].count().reset_index()
    products_meta.columns = ["urun url", "count"]
    products_meta = products_meta.sort_values("count", ascending=False)

    index_list = []
    for _, prow in products_meta.iterrows():
        url  = prow["urun url"]
        name = df[df["urun url"] == url]["urun adi"].iloc[0] if "urun adi" in df.columns else url
        prod_df  = df[df["urun url"] == url].copy()
        prod_df.reset_index(drop=True, inplace=True)

        overall   = float(prod_df["puan"].mean())
        n_reviews = len(prod_df)

        # ── Keyword tabanlı multi-label aspect ataması ──
        # Her yorum, BERTopic'in hard topic ataması yerine aspect keyword'leriyle
        # eşleştirilerek birden fazla aspect'e katkıda bulunabilir.
        # Örnek: "Kumaşı güzel ama kalıbı dar" → hem Kumaş hem Kalıp aspect'ine dahil.
        merged: dict[str, dict] = {
            label: {
                "indices": [],
                "terms": topic_terms.get(label_to_tid.get(label, -1), [])[:8],
            }
            for label in aspect_kws
        }
        for local_idx, (_, r) in enumerate(prod_df.iterrows()):
            text_lower = tr_lower(clean(r["temiz urun yorum"]))
            # Kısa yorumlar (< 60 karakter): 1 keyword yeterli
            # Uzun yorumlar: en az 2 farklı keyword gerekir → yanlış eşleşme azalır
            min_kw = 1 if len(text_lower) < 60 else 2
            matched_aspects = _assign_review_to_aspects(text_lower, aspect_kws, min_matches=min_kw)
            for label in matched_aspects:
                if label in merged:
                    merged[label]["indices"].append(local_idx)

        # AspectResult listesi oluştur
        aspect_results: list[AspectResult] = []
        for label, data in sorted(merged.items(), key=lambda x: -len(x[1]["indices"])):
            idxs = data["indices"]
            if len(idxs) < 3:  # çok az yorumlu aspect'ler = gürültü, atla
                continue
            ar = compute_aspect(prod_df, label, idxs, data["terms"])
            aspect_results.append(ar)

        # Yorum sayısına göre sırala, en fazla 10 aspect
        aspect_results.sort(key=lambda a: -a.review_count)
        aspect_results = aspect_results[:10]

        summary = generate_summary(name, aspect_results, overall, n_reviews)
        score_dist = {str(i): int((prod_df["puan"] == i).sum()) for i in range(1, 6)}
        top_reviews = (
            prod_df["temiz urun yorum"]
            .sort_values(key=lambda x: x.str.len(), ascending=False)
            .head(5)
            .tolist()
        )

        result = {
            "product_id": url_to_id(url),
            "product_name": name,
            "product_url": url,
            "category": category,
            "category_label": CATEGORY_LABELS.get(category, category),
            "total_reviews": n_reviews,
            "overall_score": round(overall, 2),
            "overall_sentiment": score_to_sentiment(overall),
            "stars": stars(overall),
            "summary": summary,
            "score_distribution": score_dist,
            "aspects": [asdict(a) for a in aspect_results],
            "top_reviews": top_reviews,
            "generated_at": "2026-04-06",
        }

        pid = result["product_id"]
        (cat_dir / f"{pid}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        index_list.append({
            "product_id": pid,
            "product_name": name[:80],
            "product_url": url,
            "total_reviews": n_reviews,
            "overall_score": round(overall, 2),
            "overall_sentiment": score_to_sentiment(overall),
            "aspect_count": len(aspect_results),
        })
        print(f"    ✓ {name[:50]} ({n_reviews} yorum, {len(aspect_results)} aspect)")

    # Kategori index
    (cat_dir / "index.json").write_text(
        json.dumps({"category": category, "products": index_list}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return index_list

# ─── CLI ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=WEB_DATA_DIR)
    parser.add_argument("--n-topics", type=int, default=12, help="BERTopic hedef topic sayısı")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cats = CATEGORIES if args.all else ([args.category] if args.category else ["mouse"])

    global_index: dict[str, Any] = {}
    for cat in cats:
        print(f"\n[{cat}] başlıyor...")
        idx = analyze_category_bertopic(cat, args.output_dir, n_topics=args.n_topics)
        global_index[cat] = {
            "category_label": CATEGORY_LABELS.get(cat, cat),
            "product_count": len(idx),
            "products": idx,
        }

    g_path = args.output_dir / "global_index.json"
    g_path.write_text(json.dumps(global_index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ Global index: {g_path}")

if __name__ == "__main__":
    main()
