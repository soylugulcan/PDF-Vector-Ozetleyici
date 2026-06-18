# build_vector_index.py

import os

from services.local_dataset_loader import (
    load_saved_govreport_dataset,
    validate_documents
)
from services.text_splitter import split_text
from services.embedding_service import EmbeddingService
from services.vector_store import FaissVectorStore


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASET_JSON_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "govreport_sample",
    "govreport_sample.json"
)

INDEX_OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "../vectorstore",
    "faiss_index"
)


def build_index_from_saved_dataset(
    dataset_json_path=DATASET_JSON_PATH,
    index_output_path=INDEX_OUTPUT_PATH,
    chunk_size=1200,
    overlap=200
):
    print("=" * 70)
    print("1. Yerel dataset okunuyor")
    print("=" * 70)

    print(f"Okunacak dataset yolu: {dataset_json_path}")
    print(f"Index kayıt yolu: {index_output_path}")

    documents = load_saved_govreport_dataset(dataset_json_path)
    documents = validate_documents(documents)

    print("=" * 70)
    print("2. Dokümanlar parçalara bölünüyor")
    print("=" * 70)

    all_chunks = []
    all_metadata = []

    for doc_index, doc in enumerate(documents):
        chunks = split_text(
            text=doc["text"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk_index, chunk in enumerate(chunks):
            all_chunks.append(chunk)

            all_metadata.append({
                "doc_id": doc["id"],
                "source": doc["source"],
                "chunk_index": chunk_index,
                "chunk_text": chunk,
                "reference_summary": doc["summary"],
                "original_document_index": doc_index,
                "chunk_char_length": len(chunk)
            })

    print(f"Toplam doküman sayısı: {len(documents)}")
    print(f"Toplam parça sayısı: {len(all_chunks)}")

    if not all_chunks:
        raise ValueError("Hiç metin parçası üretilemedi. Dataset içeriğini kontrol et.")

    print("=" * 70)
    print("3. Embeddingler üretiliyor")
    print("=" * 70)

    embedding_service = EmbeddingService(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    embeddings = embedding_service.embed_texts(
        texts=all_chunks,
        batch_size=32
    )

    dimension = embeddings.shape[1]

    print(f"Embedding boyutu: {dimension}")

    print("=" * 70)
    print("4. FAISS index oluşturuluyor")
    print("=" * 70)

    vector_store = FaissVectorStore(
        dimension=dimension,
        index_path=index_output_path
    )

    vector_store.add_documents(
        embeddings=embeddings,
        metadatas=all_metadata
    )

    print("=" * 70)
    print("5. FAISS index diske kaydediliyor")
    print("=" * 70)

    vector_store.save()

    print("=" * 70)
    print("İşlem tamamlandı")
    print("=" * 70)

    print("Oluşması gereken dosyalar:")
    print(os.path.join(index_output_path, "index.faiss"))
    print(os.path.join(index_output_path, "metadata.pkl"))


if __name__ == "__main__":
    build_index_from_saved_dataset()