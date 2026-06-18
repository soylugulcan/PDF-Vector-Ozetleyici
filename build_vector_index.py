# services/dataset_loader.py

from datasets import load_dataset
import os
import json


# Bu dosyanın bulunduğu yer:
# C:\Users\musta\PycharmProjects\PDFAnaliz\services\dataset_loader.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Bir üst klasör proje ana klasörü:
# C:\Users\musta\PycharmProjects\PDFAnaliz
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Datasetin kesin kaydedileceği klasör:
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "govreport_sample"
)


def load_govreport_sample(sample_size=100):
    """
    GovReport veri kümesinden küçük bir örnek alır.
    """

    print("GovReport dataset yükleniyor...")

    dataset = load_dataset(
        "ccdv/govreport-summarization",
        split="train"
    )

    print(f"Toplam train kayıt sayısı: {len(dataset)}")

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


def save_documents_as_json(documents, output_dir=DEFAULT_OUTPUT_DIR):
    """
    Çekilen GovReport örneklerini tek JSON dosyasına kaydeder.
    """

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


def save_dataset_metadata(documents, output_dir=DEFAULT_OUTPUT_DIR):
    """
    Dataset hakkında metadata dosyası oluşturur.
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


def save_documents_as_txt(documents, output_dir=None):
    """
    Her dokümanı ayrı TXT dosyası olarak kaydeder.
    """

    if output_dir is None:
        output_dir = os.path.join(DEFAULT_OUTPUT_DIR, "txt_files")

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


def prepare_and_save_govreport_dataset(
    sample_size=100,
    output_dir=DEFAULT_OUTPUT_DIR,
    save_txt=False
):
    """
    Dataset'i indirir ve proje kök klasörü altındaki data klasörüne kaydeder.
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
    print(f"Proje ana klasörü: {PROJECT_ROOT}")
    print(f"Dataset kayıt klasörü: {DEFAULT_OUTPUT_DIR}")

    prepare_and_save_govreport_dataset(
        sample_size=100,
        output_dir=DEFAULT_OUTPUT_DIR,
        save_txt=False
    )

    print("İşlem tamamlandı.")