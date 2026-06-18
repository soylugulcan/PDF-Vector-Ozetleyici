# services/pdf_reader.py

import fitz  # PyMuPDF


def extract_text_from_pdf(uploaded_file):
    """
    Streamlit üzerinden yüklenen PDF dosyasından metin çıkarır.

    uploaded_file:
        Streamlit file_uploader ile gelen dosya nesnesidir.

    Bu fonksiyon:
    - PDF'i byte olarak okur
    - PyMuPDF ile açar
    - Sayfa sayfa metin çıkarır
    - Sayfa numaralarını metne ekler

    Not:
    Eğer PDF taranmış görsel PDF ise bu yöntem metin çıkaramayabilir.
    O durumda OCR gerekir.
    """

    pdf_bytes = uploaded_file.read()

    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    full_text = ""

    for page_index, page in enumerate(doc):
        page_text = page.get_text()

        full_text += f"\n\n--- Sayfa {page_index + 1} ---\n"
        full_text += page_text

    return full_text.strip()


def clean_pdf_text(text):
    """
    PDF'ten gelen metni basit şekilde temizler.

    Burada çok agresif temizlik yapmıyoruz.
    Çünkü özetleme modeli bağlamı kaybetmemeli.
    """

    if not text:
        return ""

    # Çok fazla boş satırı azaltıyoruz.
    lines = text.splitlines()

    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)