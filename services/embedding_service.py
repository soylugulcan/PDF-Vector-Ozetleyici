# services/embedding_service.py

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    Metinleri embedding vektörlerine dönüştüren sınıf.

    Burada dış yapay zekâ API'si kullanmıyoruz.
    Model bilgisayara indirilir ve lokal çalışır.

    İlk çalıştırmada model internetten indirilebilir.
    Sonraki çalıştırmalarda cache üzerinden açılır.
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        model_name:
            Kullanılacak embedding modelinin adı.

        all-MiniLM-L6-v2:
            Hafif, hızlı ve başlangıç için idealdir.

        Türkçe/çok dilli veri için ileride şu modele geçebiliriz:
            sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
        """

        print(f"Embedding modeli yükleniyor: {model_name}")

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        print("Embedding modeli hazır.")

    def embed_texts(self, texts, batch_size=32):
        """
        Birden fazla metni embedding vektörüne dönüştürür.

        texts:
            Liste halinde metin parçaları.

        batch_size:
            Aynı anda kaç metnin işleneceği.
            RAM düşükse 16 veya 8 yapılabilir.
        """

        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        return embeddings

    def embed_query(self, query):
        """
        Tek bir sorguyu embedding vektörüne dönüştürür.

        Vector database içinde benzer metin ararken kullanacağız.
        """

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding