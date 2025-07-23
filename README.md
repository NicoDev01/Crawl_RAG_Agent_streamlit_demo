# Crawl RAG Agent - Streamlit Demo

Ein vollständiges RAG (Retrieval-Augmented Generation) System mit verteilter Cloud-Architektur.

## 🚀 Live Demo

[Streamlit Community Cloud App](https://your-app-url.streamlit.app) (wird nach Deployment verfügbar)

## 🏗️ Architektur

- **Frontend**: Streamlit Community Cloud
- **Crawling**: Modal.com Serverless Service
- **Vektordatenbank**: ChromaDB (In-Memory)
- **Embeddings**: ChromaDB Standard oder Google Vertex AI

## ✨ Features

- 🕷️ **Web Crawling**: Einzelne URLs, Sitemaps, rekursives Crawling
- 📚 **Wissensdatenbank**: Automatische Erstellung aus gecrawlten Inhalten
- 🤖 **RAG Chat**: Fragen gegen die Wissensdatenbank stellen
- 🔄 **Real-time**: Live Progress-Tracking und Session Management
- 🌐 **Cloud-Native**: Vollständig serverlos und skalierbar

## 🛠️ Lokale Entwicklung

1. Repository klonen:
```bash
git clone https://github.com/NicoDev01/Crawl_RAG_Agent_streamlit_demo.git
cd Crawl_RAG_Agent_streamlit_demo
```

2. Dependencies installieren:
```bash
pip install -r requirements.txt
```

3. Streamlit App starten:
```bash
streamlit run streamlit_app.py
```

## ⚙️ Konfiguration

### Erforderliche Secrets (Streamlit Cloud)

```toml
# Modal.com API (ERFORDERLICH)
MODAL_API_URL = "https://nico-gt91--crawl4ai-service"
MODAL_API_KEY = "your-modal-api-key"
```

### Optionale Secrets (für erweiterte Features)

```toml
# Google Cloud (für Vertex AI Embeddings)
GOOGLE_CLOUD_PROJECT = "your-gcp-project-id"
GOOGLE_CLOUD_LOCATION = "us-central1"
GOOGLE_APPLICATION_CREDENTIALS_JSON = "base64-encoded-service-account"

# OpenAI (für erweiterte RAG Features)
OPENAI_API_KEY = "your-openai-api-key"
```

## 📖 Verwendung

1. **Crawler Test**: Teste das Web-Crawling mit verschiedenen URLs
2. **Wissensdatenbank erstellen**: Crawle Webseiten und erstelle eine durchsuchbare Datenbank
3. **RAG Chat**: Stelle Fragen gegen die erstellte Wissensdatenbank

## 🔧 Technische Details

- **Crawling Service**: Modal.com mit Crawl4AI und Playwright
- **Text Processing**: Intelligentes Chunking nach Markdown-Headern
- **Embeddings**: ChromaDB Standard oder Vertex AI Multilingual
- **Vector Search**: ChromaDB mit Cosine Similarity
- **UI Framework**: Streamlit mit Session State Management

## 📝 Lizenz

MIT License

## 🤝 Beitragen

Pull Requests sind willkommen! Für größere Änderungen öffne bitte zuerst ein Issue.

## 📞 Support

Bei Fragen oder Problemen öffne ein GitHub Issue.