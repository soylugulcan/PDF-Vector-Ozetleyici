
```markdown
#  RAG Tabanlı PDF Özetleme ve Soru-Cevap Sistemi

Bu proje; Doğal Dil İşleme (NLP), Büyük Dil Modelleri (LLM) ve Retrieval-Augmented Generation (RAG) tekniklerini kullanarak PDF dokümanlarının otomatik özetlenmesini ve doküman üzerinden akıllı analizler yapılmasını amaçlamaktadır.

---

##  Özellikler

### PDF İşleme
* PDF dosyasından metin çıkarma
* Metin temizleme
* Çok sayfalı doküman desteği

### Metin Bölme (Chunking)
* Uzun metinleri küçük parçalara ayırma
* Overlap (örtüşme) kullanarak bağlam kaybını azaltma

### Embedding Üretimi
* Sentence Transformers kullanımı
* Metinlerin vektör uzayına dönüştürülmesi

### Vektör Arama
* FAISS ile yüksek hızlı benzerlik araması
* En alakalı içeriklerin bulunması

### Yapay Zeka Destekli Özetleme
* Ollama entegrasyonu
* Qwen / Llama / Mistral desteği
* Tamamen yerel (local) LLM çalıştırabilme

### Kullanıcı Arayüzü
* Streamlit tabanlı web arayüzü
* Kolay PDF yükleme
* Anlık sonuç görüntüleme

---

#  Sistem Mimarisi

```text
       [ PDF Dokümanı ]
              │
              ▼
      [ Text Extraction ]
              │
              ▼
       [ Text Cleaning ]
              │
              ▼
          [ Chunking ]
              │
              ▼
    [ Embedding Generation ]
              │
              ▼
       [ FAISS Indexing ]
              │
              ▼
  [ Relevant Context Retrieval ]
              │
              ▼
  [ LLM (Qwen / Llama / Mistral) ]
              │
              ▼
     [ Summary Generation ]

```

---

#  Proje Yapısı

```text
├──  outputs/                   
│   │                 
│   └── templates/               
│
├── services/                     
│   ├── data/                     
│   ├── outputs/                 
│   ├── build_vector_index.py     
│   ├── dataset_loader.py         
│   ├── embedding_service.py      
│   ├── local_dataset_loader.py  
│   ├── ollama_summarizer.py      
│   ├── pdf_reader.py            
│   ├── summarizer.py            
│   ├── text_splitter.py          
│   └── vector_store.py           
│
├── vector_diagnostics/           
├── vectorstore/                  
│
├── main.py                       
├── rapor.py                      
├── uretici.py                    
├── test_vector_search.py         
└──  requirements.txt              

```

---

# Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın

```bash
git clone [https://github.com/kullanici/proje.git](https://github.com/kullanici/proje.git)
cd proje

```

### 2. Sanal Ortam Oluşturun

```bash
python -m venv .venv

```

**Sanal Ortamı Aktifleştirme:**

* **Windows:** `.venv\Scripts\activate`
* **Linux / Mac:** `source .venv/bin/activate`

### 3. Gereksinimleri Kurun

```bash
pip install -r requirements.txt

```

### 4. Ollama Kurulumu

1. Önce bilgisayarınıza Ollama'yı indirin ve kurun: [ollama.com](https://ollama.com)
2. Tercih ettiğiniz modellerden birini terminal üzerinden bilgisayarınıza çekin:

```bash
ollama pull qwen2.5:3b
# veya
ollama pull llama3

```

### 5. Uygulamayı Çalıştırın

```bash
streamlit run app.py

```

---

# Kullanılan Modeller

* **Embedding Modeli:** `all-MiniLM-L6-v2` (Sentence Transformers)
* **Üretici Yapay Zeka Modelleri (LLM):** `Qwen`, `Llama` veya `Mistral` (Ollama aracılığıyla lokalde çalıştırılır)

```

```


<img width="1903" height="888" alt="Ekran görüntüsü 2026-06-19 080134" src="https://github.com/user-attachments/assets/ebb28b16-d13d-4ef9-85e7-daaef2328dec" />
<img width="1899" height="895" alt="Ekran görüntüsü 2026-06-19 080247" src="https://github.com/user-attachments/assets/f2b89351-8ae8-4d9b-826b-958d0e1ee186" />
<img width="1895" height="873" alt="Ekran görüntüsü 2026-06-19 080300" src="https://github.com/user-attachments/assets/1bf69fa2-4f9b-417c-87b7-78dc55547476" />
