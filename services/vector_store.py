# services/vector_store.py

import os
import pickle

import faiss
import numpy as np


class FaissVectorStore:
    """
    FAISS tabanlı vector database sınıfı.

    Görevleri:
    - Embeddingleri FAISS index içine eklemek
    - Metadata bilgilerini saklamak
    - Index ve metadata dosyalarını diske kaydetmek
    - Kaydedilen index'i tekrar yüklemek
    - Benzerlik araması yapmak
    """

    def __init__(self, dimension, index_path):
        self.dimension = dimension

        # Gelen path'i absolute path'e çeviriyoruz.
        # Böylece PyCharm / Streamlit çalışma klasörü değişse bile yol bozulmaz.
        self.index_path = os.path.abspath(index_path)

        # Embeddingleri normalize ettiğimiz için cosine similarity benzeri arama yapar.
        self.index = faiss.IndexFlatIP(dimension)

        self.metadata = []

    def add_documents(self, embeddings, metadatas):
        """
        Embeddingleri FAISS index içine ekler.
        """

        if len(embeddings) != len(metadatas):
            raise ValueError(
                f"Embedding sayısı ile metadata sayısı eşit değil. "
                f"Embedding: {len(embeddings)}, Metadata: {len(metadatas)}"
            )

        embeddings = np.array(embeddings).astype("float32")

        self.index.add(embeddings)
        self.metadata.extend(metadatas)

        print(f"FAISS index içine eklenen kayıt sayısı: {len(metadatas)}")
        print(f"FAISS toplam kayıt sayısı: {self.index.ntotal}")

    def search(self, query_embedding, top_k=5):
        """
        Sorguya en yakın metin parçalarını döndürür.
        """

        if self.index.ntotal == 0:
            return []

        query_embedding = np.array(query_embedding).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue

            results.append({
                "score": float(score),
                "metadata": self.metadata[idx]
            })

        return results

    def save(self):
        """
        FAISS index ve metadata dosyasını diske kaydeder.

        Oluşacak dosyalar:
        - index.faiss
        - metadata.pkl
        """

        print("FAISS kayıt klasörü:")
        print(self.index_path)

        os.makedirs(self.index_path, exist_ok=True)

        index_file = os.path.join(self.index_path, "index.faiss")
        metadata_file = os.path.join(self.index_path, "metadata.pkl")

        print("Kaydedilecek index dosyası:")
        print(index_file)

        print("Kaydedilecek metadata dosyası:")
        print(metadata_file)

        faiss.write_index(self.index, index_file)

        with open(metadata_file, "wb") as file:
            pickle.dump(self.metadata, file)

        print("FAISS index kaydetme işlemi çağrıldı.")
        print("Metadata kaydetme işlemi çağrıldı.")

        # Kaydettikten sonra gerçekten oluştu mu kontrol ediyoruz.
        if not os.path.exists(index_file):
            raise FileNotFoundError(
                f"FAISS index dosyası kaydedilemedi: {index_file}"
            )

        if not os.path.exists(metadata_file):
            raise FileNotFoundError(
                f"Metadata dosyası kaydedilemedi: {metadata_file}"
            )

        print("Dosyalar başarıyla oluşturuldu:")
        print(index_file)
        print(metadata_file)

        print(f"index.faiss boyutu: {os.path.getsize(index_file)} byte")
        print(f"metadata.pkl boyutu: {os.path.getsize(metadata_file)} byte")

    def load(self):
        """
        Kaydedilmiş FAISS index ve metadata dosyasını yükler.
        """

        index_file = os.path.join(self.index_path, "index.faiss")
        metadata_file = os.path.join(self.index_path, "metadata.pkl")

        print("Yüklenecek FAISS index dosyası:")
        print(index_file)

        print("Yüklenecek metadata dosyası:")
        print(metadata_file)

        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index dosyası bulunamadı: {index_file}")

        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata dosyası bulunamadı: {metadata_file}")

        self.index = faiss.read_index(index_file)

        with open(metadata_file, "rb") as file:
            self.metadata = pickle.load(file)

        print("FAISS index başarıyla yüklendi.")
        print(f"Toplam index kaydı: {self.index.ntotal}")
        print(f"Toplam metadata kaydı: {len(self.metadata)}")