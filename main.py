# app.py

import os
import streamlit as st

from services.pdf_reader import extract_text_from_pdf, clean_pdf_text
from services.text_splitter import split_text
from services.embedding_service import EmbeddingService
from services.vector_store import FaissVectorStore
from services.ollama_summarizer import OllamaSummarizer


# =========================================================
# PROJE ANA KLASÖRÜ
# =========================================================
# Bu satır sayesinde PyCharm projeyi hangi klasörden çalıştırırsa çalıştırsın
# FAISS index dosyalarının yolu doğru bulunur.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

VECTOR_INDEX_PATH = os.path.join(
    PROJECT_ROOT,
    "vectorstore",
    "faiss_index"
)

INDEX_FILE_PATH = os.path.join(VECTOR_INDEX_PATH, "index.faiss")
METADATA_FILE_PATH = os.path.join(VECTOR_INDEX_PATH, "metadata.pkl")


# =========================================================
# STREAMLIT SAYFA AYARLARI
# =========================================================
st.set_page_config(
    page_title="PDF Vector Özetleyici",
    layout="wide"
)


st.title("PDF Vector Tabanlı Özetleyici")

st.write("""
Bu uygulama PDF içeriğini okur, metni parçalara böler,
kendi oluşturduğumuz FAISS vector database üzerinden benzer içerikleri bulur
ve Ollama üzerinde çalışan yerel model ile Türkçe özet üretir.
""")


# =========================================================
# SIDEBAR AYARLARI
# =========================================================
with st.sidebar:
    st.header("Ayarlar")

    model_name = st.selectbox(
        "Ollama modeli",
        [
            "qwen2.5:3b",
            "qwen2.5:7b",
            "mistral",
            "llama3.1"
        ],
        index=0
    )

    chunk_size = st.slider(
        "PDF parça boyutu",
        min_value=800,
        max_value=3000,
        value=1200,
        step=200
    )

    overlap = st.slider(
        "Parça bindirme miktarı",
        min_value=0,
        max_value=600,
        value=200,
        step=50
    )

    max_search_chunks = st.slider(
        "Vector arama yapılacak PDF parça sayısı",
        min_value=1,
        max_value=30,
        value=10,
        step=1
    )

    top_k = st.slider(
        "Her PDF parçası için getirilecek benzer parça sayısı",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )

    max_pdf_chars_for_summary = st.slider(
        "Özete gönderilecek maksimum PDF karakteri",
        min_value=3000,
        max_value=20000,
        value=8000,
        step=1000
    )

    max_context_chars_for_summary = st.slider(
        "Özete gönderilecek maksimum vector bağlam karakteri",
        min_value=3000,
        max_value=20000,
        value=8000,
        step=1000
    )

    st.info("""
PDF çok uzunsa daha hızlı sonuç için:
- Vector arama yapılacak parça sayısını azaltabilirsin.
- Daha küçük Ollama modeli seçebilirsin.
""")


# =========================================================
# FAISS INDEX KONTROLÜ
# =========================================================
st.subheader("Sistem Kontrolü")

col1, col2 = st.columns(2)

with col1:
    st.write("FAISS index yolu:")
    st.code(VECTOR_INDEX_PATH)

with col2:
    if os.path.exists(INDEX_FILE_PATH) and os.path.exists(METADATA_FILE_PATH):
        st.success("FAISS index dosyaları bulundu.")
    else:
        st.error("FAISS index dosyaları bulunamadı.")

        st.write("Eksik olması muhtemel dosyalar:")

        st.code(INDEX_FILE_PATH)
        st.code(METADATA_FILE_PATH)

        st.warning("""
Önce proje ana klasöründe şu komutu çalıştırmalısın:

python build_vector_index.py
""")

        st.stop()


# =========================================================
# PDF YÜKLEME
# =========================================================
uploaded_file = st.file_uploader(
    "PDF dosyası yükle",
    type=["pdf"]
)


if uploaded_file is not None:
    st.success(f"PDF yüklendi: {uploaded_file.name}")

    if st.button("PDF'i Özetle"):
        # =========================================================
        # 1. PDF METNİNİ ÇIKAR
        # =========================================================
        with st.spinner("PDF metni çıkarılıyor..."):
            raw_pdf_text = extract_text_from_pdf(uploaded_file)
            pdf_text = clean_pdf_text(raw_pdf_text)

        if not pdf_text:
            st.error("""
PDF'ten metin çıkarılamadı.

Bu PDF büyük ihtimalle taranmış/görsel PDF olabilir.
Bu durumda OCR eklemek gerekir.
""")
            st.stop()

        st.subheader("PDF'ten Çıkan Metin Önizleme")

        st.text_area(
            "İlk metin",
            pdf_text[:3000],
            height=250
        )

        # =========================================================
        # 2. PDF METNİNİ PARÇALARA BÖL
        # =========================================================
        with st.spinner("PDF parçalanıyor..."):
            chunks = split_text(
                text=pdf_text,
                chunk_size=chunk_size,
                overlap=overlap
            )

        st.write(f"Toplam PDF parça sayısı: {len(chunks)}")

        if not chunks:
            st.error("PDF metni parçalara bölünemedi.")
            st.stop()

        # =========================================================
        # 3. EMBEDDING MODELİNİ YÜKLE
        # =========================================================
        with st.spinner("Embedding modeli yükleniyor..."):
            embedding_service = EmbeddingService(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

        # =========================================================
        # 4. FAISS INDEX YÜKLE
        # =========================================================
        with st.spinner("FAISS index yükleniyor..."):
            dummy_embedding = embedding_service.embed_query("test")
            dimension = dummy_embedding.shape[1]

            vector_store = FaissVectorStore(
                dimension=dimension,
                index_path=VECTOR_INDEX_PATH
            )

            vector_store.load()

        # =========================================================
        # 5. PDF PARÇALARI İLE VECTOR DATABASE ARAMASI YAP
        # =========================================================
        all_contexts = []
        seen_contexts = set()

        with st.spinner("PDF parçaları vector database ile karşılaştırılıyor..."):
            selected_chunks = chunks[:max_search_chunks]

            for chunk in selected_chunks:
                query_embedding = embedding_service.embed_query(chunk)

                results = vector_store.search(
                    query_embedding=query_embedding,
                    top_k=top_k
                )

                for result in results:
                    metadata = result.get("metadata", {})

                    # build_vector_index.py içinde metadata alanını "chunk_text" olarak kaydetmiştik.
                    context_text = metadata.get("chunk_text", "")

                    if context_text and context_text not in seen_contexts:
                        seen_contexts.add(context_text)
                        all_contexts.append(context_text)

        st.subheader("Vector Database'den Gelen Benzer Bağlamlar")

        st.write(f"Bulunan benzersiz bağlam sayısı: {len(all_contexts)}")

        with st.expander("Benzer bağlamları göster"):
            if all_contexts:
                for i, context in enumerate(all_contexts[:10], start=1):
                    st.markdown(f"### Bağlam {i}")
                    st.write(context[:1200])
            else:
                st.warning("Benzer bağlam bulunamadı. Yine de PDF metni üzerinden özet üretilebilir.")

        # =========================================================
        # 6. ÖZETLEME İÇİN METNİ HAZIRLA
        # =========================================================
        context_text = "\n\n".join(all_contexts)

        final_input = f"""
Kullanıcının PDF'inden çıkarılan metin:
{pdf_text[:max_pdf_chars_for_summary]}

Vector database üzerinden bulunan benzer referans parçalar:
{context_text[:max_context_chars_for_summary]}
"""

        # =========================================================
        # 7. OLLAMA İLE ÖZET ÜRET
        # =========================================================
        st.subheader("Üretilen Özet")

        try:
            with st.spinner("Yerel model ile özet üretiliyor..."):
                summarizer = OllamaSummarizer(
                    model_name=model_name
                )

                summary = summarizer.summarize(final_input)

            st.markdown(summary)

            # =====================================================
            # 8. ÖZETİ TXT OLARAK İNDİR
            # =====================================================
            st.download_button(
                label="Özeti TXT olarak indir",
                data=summary,
                file_name="pdf_ozeti.txt",
                mime="text/plain"
            )

        except ConnectionError as error:
            st.error(str(error))

            st.warning("""
Ollama çalışmıyor olabilir.

Terminalde şunu çalıştır:

ollama serve

Sonra başka terminalde modeli test et:

ollama run qwen2.5:7b
""")

        except TimeoutError as error:
            st.error(str(error))

            st.warning("""
Model cevap verirken zaman aşımına uğradı.

Çözüm olarak:
- qwen2.5:3b modelini seçebilirsin.
- PDF karakter limitini düşürebilirsin.
- Vector bağlam karakter limitini düşürebilirsin.
""")

        except Exception as error:
            st.error("Beklenmeyen bir hata oluştu.")
            st.exception(error)