# services/ollama_summarizer.py

import requests


class OllamaSummarizer:
    """
    Ollama üzerinden yerel model ile özetleme yapan sınıf.
    """

    def __init__(
        self,
        model_name="qwen2.5:3b",
        api_url="http://localhost:11434/api/generate"
    ):
        self.model_name = model_name
        self.api_url = api_url

    def summarize(self, text):
        """
        Verilen metni Türkçe özetler.
        """

        prompt = f"""
Sen profesyonel bir PDF analiz ve özetleme asistanısın.

Aşağıdaki metni Türkçe olarak özetle.

Kurallar:
1. Önce kısa bir genel özet yaz.
2. Sonra ana noktaları madde madde çıkar.
3. Önemli kavramları açıkla.
4. Metinde sonuç, öneri, karar veya bulgu varsa ayrıca belirt.
5. Gereksiz tekrarları çıkar.
6. Uydurma bilgi ekleme.
7. Sadece verilen metne dayan.

Cevap formatı:

# Genel Özet

# Ana Noktalar

# Önemli Kavramlar

# Sonuç / Değerlendirme

Metin:
{text}
"""

        return self._generate(prompt)

    def _generate(self, prompt):
        """
        Ollama generate endpoint'ine istek gönderir.
        """

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=600
            )

            # Eğer Ollama 404, 500 vb. dönerse detaylı mesaj göstermek için kontrol ediyoruz.
            if response.status_code != 200:
                raise Exception(
                    f"Ollama API hata verdi.\n\n"
                    f"Status Code: {response.status_code}\n"
                    f"URL: {self.api_url}\n"
                    f"Model: {self.model_name}\n"
                    f"Cevap: {response.text}\n\n"
                    f"Muhtemel çözüm:\n"
                    f"1. Terminalde 'ollama list' çalıştır.\n"
                    f"2. Seçtiğin model listede yoksa indir:\n"
                    f"   ollama pull {self.model_name}\n"
                    f"3. Sonra test et:\n"
                    f"   ollama run {self.model_name}"
                )

            data = response.json()

            return data.get("response", "").strip()

        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Ollama servisine bağlanılamadı.\n\n"
                "Çözüm:\n"
                "1. Ollama kurulu mu kontrol et: ollama --version\n"
                "2. Terminalde çalıştır: ollama serve\n"
                "3. Modeli test et: ollama run qwen2.5:3b"
            )

        except requests.exceptions.Timeout:
            raise TimeoutError(
                "Ollama cevap verirken zaman aşımına uğradı.\n\n"
                "Çözüm:\n"
                "- Daha küçük model seç: qwen2.5:3b\n"
                "- PDF karakter limitini düşür\n"
                "- Vector bağlam karakter limitini düşür"
            )