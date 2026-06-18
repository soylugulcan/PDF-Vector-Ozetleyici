# test_vector_search.py

from services.embedding_service import EmbeddingService
from services.vector_store import FaissVectorStore


def main():
    """
    Oluşturduğumuz FAISS index üzerinde test araması yapar.

    Bu dosyanın amacı:
    - index.faiss düzgün oluşmuş mu?
    - metadata.pkl düzgün yükleniyor mu?
    - Arama sonucu mantıklı parçalar dönüyor mu?

    kontrol etmektir.
    """

    query = """
    What are the main findings and recommendations in the government report?
    """

    print("Embedding modeli yükleniyor...")

    embedding_service = EmbeddingService(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    query_embedding = embedding_service.embed_query(query)

    dimension = query_embedding.shape[1]

    print("FAISS index yükleniyor...")

    vector_store = FaissVectorStore(
        dimension=dimension,
        index_path="vectorstore/faiss_index"
    )

    vector_store.load()

    print("Arama yapılıyor...")

    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=5
    )

    print("\nEn benzer sonuçlar:\n")

    for i, result in enumerate(results, start=1):
        metadata = result["metadata"]

        print("=" * 70)
        print(f"Sonuç {i}")
        print("=" * 70)
        print(f"Skor: {result['score']}")
        print(f"Doküman ID: {metadata['doc_id']}")
        print(f"Parça Index: {metadata['chunk_index']}")
        print("\nMetin Parçası:")
        print(metadata["chunk_text"][:1000])
        print()


if __name__ == "__main__":
    main()