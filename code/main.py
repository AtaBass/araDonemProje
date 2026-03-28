"""
Ana giriş noktası.
`python main.py` çağrısı Trendyol pipeline'ını çalıştırır:
10 kategori x 30 ürün x ürün başı 50 yorum.
"""

from pipeline import main as pipeline_main


if __name__ == "__main__":
    pipeline_main()
