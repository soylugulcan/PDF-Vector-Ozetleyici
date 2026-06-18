# services/dataset_loader.py

from datasets import load_dataset
import os
import json


def load_govreport_sample(sample_size=100):
    """
    GovReport veri kümesinden küçük bir örnek alır.

    Bu fonksiyon:
    - Hugging Face üzerinden GovReport datasetini çeker
    - Train split içinden belirtilen kadar örnek seçer
    - Her örneği Python dictionary yapısına dönüştürür

    Not:
    Hugging Face datasets kütüphanesi dataset'i kendi cache klasörüne indirir.
    Biz ayrıca aşağıdaki save fonksiyonlarıyla kendi proje klasörümüze de kaydedeceğiz.
    """

    print("GovReport dataset yükleniyor...")

    dataset = load_dataset(
        "ccdv/govreport-summarization",
        split="train"
    )

    print(f"Toplam train kayıt sayısı: {len(dataset)}")

    # Eğer istenen sample_size dataset boyutundan büyükse hata almamak için sınırlandırıyoruz.
    sample_size = min(sample_size, len(dataset))

    sample = dataset.select(range(sample_size))

    documents = []

    for i, item in enumerate(sample):
        documents.append({
            "id": f"govreport_{i}",
            "text": item["report"],
            "summary": item["summary"],
            "source": "govreport"
        })

    print(f"{len(documents)} adet kayıt hazırlandı.")

    return documents


def save_documents_as_json(documents, output_dir="data/govreport_sample"):
    """
    Çekilen GovReport örneklerini tek bir JSON dosyasına kaydeder.

    Oluşacak dosya:
    data/govreport_sample/govreport_sample.json
    """

    # Klasör yoksa oluşturur.
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "govreport_sample.json")

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"JSON dosyası kaydedildi: {json_path}")

    return json_path


def save_documents_as_txt(documents, output_dir="data/govreport_sample/txt_files"):
    """
    Her dokümanı ayrı TXT dosyası olarak kaydeder.

    Her dosyada:
    - ID
    - Source
    - Summary
    - Report Text

    bilgileri bulunur.

    Oluşacak örnek dosyalar:
    data/govreport_sample/txt_files/govreport_0.txt
    data/govreport_sample/txt_files/govreport_1.txt
    """

    os.makedirs(output_dir, exist_ok=True)

    for doc in documents:
        file_name = f"{doc['id']}.txt"
        file_path = os.path.join(output_dir, file_name)

        with open(file_path, "w", encoding="utf-8") as file:
            file.write(f"ID: {doc['id']}\n")
            file.write(f"Source: {doc['source']}\n")
            file.write("\n")
            file.write("===== SUMMARY =====\n")
            file.write(doc["summary"])
            file.write("\n\n")
            file.write("===== REPORT TEXT =====\n")
            file.write(doc["text"])

    print(f"TXT dosyaları kaydedildi: {output_dir}")

    return output_dir


def save_dataset_metadata(documents, output_dir="data/govreport_sample"):
    """
    Dataset hakkında küçük bir metadata dosyası oluşturur.

    Bu dosya kaç kayıt çekildiğini, toplam karakter sayılarını,
    ortalama metin uzunluğunu görmek için kullanışlıdır.
    """

    os.makedirs(output_dir, exist_ok=True)

    total_text_length = sum(len(doc["text"]) for doc in documents)
    total_summary_length = sum(len(doc["summary"]) for doc in documents)

    metadata = {
        "dataset_name": "ccdv/govreport-summarization",
        "sample_count": len(documents),
        "total_report_text_characters": total_text_length,
        "total_summary_characters": total_summary_length,
        "average_report_text_characters": total_text_length / len(documents) if documents else 0,
        "average_summary_characters": total_summary_length / len(documents) if documents else 0
    }

    metadata_path = os.path.join(output_dir, "metadata.json")

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"Metadata dosyası kaydedildi: {metadata_path}")

    return metadata_path


def prepare_and_save_govreport_dataset(
    sample_size=100,
    output_dir="data/govreport_sample",
    save_txt=True
):
    """
    Tüm işlemi tek fonksiyonda yapar.

    Yapılanlar:
    1. GovReport datasetinden örnek veri çeker
    2. JSON olarak kaydeder
    3. Metadata dosyası oluşturur
    4. İstenirse her dokümanı ayrı TXT dosyası olarak kaydeder

    Bu fonksiyon daha sonra build_vector_index.py içinde de kullanılabilir.
    """

    documents = load_govreport_sample(sample_size=sample_size)

    save_documents_as_json(
        documents=documents,
        output_dir=output_dir
    )

    save_dataset_metadata(
        documents=documents,
        output_dir=output_dir
    )

    if save_txt:
        save_documents_as_txt(
            documents=documents,
            output_dir=os.path.join(output_dir, "txt_files")
        )

    return documents


if __name__ == "__main__":
    """
    Bu dosyayı direkt çalıştırırsan burası çalışır.

    Terminal:
        python services/dataset_loader.py

    Çıktılar:
        data/govreport_sample/govreport_sample.json
        data/govreport_sample/metadata.json
        data/govreport_sample/txt_files/govreport_0.txt
        data/govreport_sample/txt_files/govreport_1.txt
        ...
    """

    documents = prepare_and_save_govreport_dataset(
        sample_size=100,
        output_dir="data/govreport_sample",
        save_txt=True
    )

    print("İşlem tamamlandı.")