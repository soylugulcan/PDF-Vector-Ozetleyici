# services/summarizer.py

import requests


class OllamaSummarizer:
    """
    Ollama üzerinden yerel model ile özetleme yapar.

    Bu yapı dış AI API kullanmaz.
    Ollama bilgisayarda çalışıyorsa localhost üzerinden cevap alır.
    """

    def __init__(self, model_name="qwen2.5:7b"):
        self.model_name = model_name
        self.url = "http://localhost:11434/api/generate"

    def summarize(self, text):
        prompt = f"""
Aşağıdaki metni Türkçe olarak özetle.

Kurallar:
- Gereksiz tekrarları çıkar.
- Ana fikirleri koru.
- Madde madde ve anlaşılır yaz.
- Teknik terimleri mümkün olduğunca açıkla.
- Çok uzun cevap verme.

Metin:
{text}
"""

        response = requests.post(
            self.url,
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=300
        )

        response.raise_for_status()

        return response.json()["response"]