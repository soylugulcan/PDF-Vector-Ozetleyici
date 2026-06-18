# services/text_splitter.py


def split_text(text, chunk_size=1200, overlap=200):
    """
    Uzun metni küçük parçalara böler.

    Neden parçalama yapıyoruz?

    Çünkü:
    - Uzun PDF/rapor metinleri tek seferde modele verilemez.
    - Embedding modelleri çok uzun metinlerde verimsiz çalışır.
    - Vector database araması parça bazlı daha doğru sonuç verir.

    chunk_size:
        Her parçanın yaklaşık karakter uzunluğu.

    overlap:
        İki parça arasında tekrar eden karakter miktarı.
        Bu sayede konu bir parçadan diğerine taşarken anlam kaybı azalır.
    """

    if text is None:
        return []

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks