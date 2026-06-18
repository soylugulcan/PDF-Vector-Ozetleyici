# services/local_dataset_loader.py

import json
import os


def load_saved_govreport_dataset(
    json_path="data/govreport_sample/govreport_sample.json"
):
    """
    Daha önce kaydettiğimiz GovReport örnek datasetini JSON dosyasından okur.

    Bu fonksiyon Hugging Face'e bağlanmaz.
    İnternetten veri çekmez.
    Sadece proje klasöründeki JSON dosyasını okur.

    Beklenen JSON formatı:
    [
        {
            "id": "govreport_0",
            "text": "...",
            "summary": "...",
            "source": "govreport"
        },
        ...
    ]
    """

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Dataset JSON dosyası bulunamadı: {json_path}\n"
            f"Önce services/dataset_loader.py dosyasını çalıştırıp dataset'i kaydetmelisin."
        )

    with open(json_path, "r", encoding="utf-8") as file:
        documents = json.load(file)

    print(f"Yerel dataset yüklendi: {json_path}")
    print(f"Toplam doküman sayısı: {len(documents)}")

    return documents


def validate_documents(documents):
    """
    JSON'dan okunan dokümanların beklenen alanlara sahip olup olmadığını kontrol eder.

    Bu kontrolü yapmamızın sebebi:
    - Eksik alan varsa vektör üretirken hata almamak
    - Veri yapısını erken aşamada doğrulamak
    """

    required_keys = ["id", "text", "summary", "source"]

    valid_documents = []

    for index, doc in enumerate(documents):
        is_valid = True

        for key in required_keys:
            if key not in doc:
                print(f"Uyarı: {index}. dokümanda '{key}' alanı eksik.")
                is_valid = False

        if is_valid and doc["text"].strip():
            valid_documents.append(doc)

    print(f"Geçerli doküman sayısı: {len(valid_documents)}")

    return valid_documents