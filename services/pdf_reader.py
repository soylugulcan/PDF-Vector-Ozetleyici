# services/pdf_reader.py

import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_file):
    """
    PDF dosyasından metin çıkarır.

    Streamlit'ten gelen uploaded_file nesnesini destekler.
    """

    pdf_bytes = pdf_file.read()

    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    full_text = ""

    for page_index, page in enumerate(doc):
        page_text = page.get_text()

        full_text += f"\n\n--- Sayfa {page_index + 1} ---\n"
        full_text += page_text

    doc.close()

    return full_text.strip()


def clean_pdf_text(text):
    """
    PDF'ten çıkarılan metni temizler.

    Amaç:
    - Gereksiz boş satırları azaltmak
    - Satır başı/sonu boşluklarını temizlemek
    - Tamamen boş satırları atmak

    Not:
    Çok agresif temizlik yapmıyoruz.
    Çünkü özetleme modelinin bağlamı kaybetmesini istemiyoruz.
    """

    if not text:
        return ""

    lines = text.splitlines()

    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)