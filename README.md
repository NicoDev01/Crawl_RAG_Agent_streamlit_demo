# 🤖 CraCha - Crawl Chat Agent

Ein intelligentes RAG (Retrieval-Augmented Generation) System mit modernster UX und verteilter Cloud-Architektur für die Erstellung von Wissensdatenbanken aus Webinhalten.

## 🚀 Live Demo

[Streamlit Community Cloud App](https://your-app-url.streamlit.app) (wird nach Deployment verfügbar)

## 🏗️ Architektur

- **Frontend**: Streamlit Community Cloud mit modernem UI/UX
- **Crawling**: Modal.com Serverless Service mit Crawl4AI
- **Vektordatenbank**: ChromaDB (In-Memory)
- **Embeddings**: Google Vertex AI Multilingual oder ChromaDB Standard
- **LLM**: Google Gemini 2.5 Flash für RAG-Antworten

## ✨ Hauptfeatures

### 🌐 Intelligentes Web-Crawling
- **Automatische URL-Typ-Erkennung**: Website, Sitemap, Einzelseite, Dokumentation
- **Real-time URL-Validierung**: Sofortige Überprüfung von Erreichbarkeit und Gültigkeit
- **Optimierte Einstellungen**: Automatische Anpassung basierend auf Website-Typ
- **Flexible Konfiguration**: Crawling-Tiefe, Seitenlimits, Chunk-Größe anpassbar

### 📚 Wissensdatenbank-Management
- **Intelligente Textaufteilung**: Optimales Chunking für präzise Suche
- **Mehrsprachige Embeddings**: Unterstützung für deutsche und internationale Inhalte
- **Automatische Optimierung**: Memory-Management und Performance-Tuning
- **Batch-Verarbeitung**: Effiziente Verarbeitung großer Datenmengen

### 💬 Erweiterte Chat-Funktionen
- **RAG-basierte Antworten**: Kontextuelle Antworten basierend auf gecrawlten Inhalten
- **Chat-Export**: Vollständige Gesprächsverläufe als Markdown exportieren
- **Session-Management**: Persistente Chat-Historie pro Wissensdatenbank
- **Typing-Indikatoren**: Moderne Chat-UX mit visuellen Feedback-Elementen

### 🎨 Moderne Benutzeroberfläche
- **Minimalistisches Design**: Fokus auf Benutzerfreundlichkeit
- **Real-time Feedback**: Sofortige Validierung und Status-Updates
- **Responsive Layout**: Optimiert für Desktop und mobile Geräte
- **Intelligente Benutzerführung**: Schritt-für-Schritt Anleitung durch den Prozess

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
# Modal.com Crawling Service (ERFORDERLICH)
MODAL_API_URL = "https://nico-gt91--crawl4ai-service"
MODAL_API_KEY = "your-modal-api-key"

# Google Cloud für Embeddings und LLM (ERFORDERLICH)
GOOGLE_CLOUD_PROJECT = "your-gcp-project-id"
GOOGLE_CLOUD_LOCATION = "us-central1"
GOOGLE_APPLICATION_CREDENTIALS_JSON = "base64-encoded-service-account-json"

# Google Gemini API für RAG-Antworten (ERFORDERLICH)
GEMINI_API_KEY = "your-gemini-api-key"
```

### Optionale Secrets (für erweiterte Features)

```toml
# Vertex AI Reranker (optional, für verbesserte Suchergebnisse)
VERTEX_RERANKER_MODEL = "text-reranking-model"

# OpenAI (optional, als Alternative zu Gemini)
OPENAI_API_KEY = "your-openai-api-key"
```

### Lokale Entwicklung (.env Datei)

```bash
# Kopiere .env.example zu .env und fülle die Werte aus
cp .env.example .env

# Erforderliche Umgebungsvariablen
MODAL_API_KEY=your-modal-api-key
GOOGLE_CLOUD_PROJECT=your-gcp-project
GEMINI_API_KEY=your-gemini-api-key
```

## 📖 Verwendung

### 1. 📚 Wissensdatenbank erstellen

1. **URL eingeben**: Gib die URL der Website ein, die du crawlen möchtest
   - Unterstützt: Websites, Sitemaps, Einzelseiten, Dokumentations-Sites
   - Real-time Validierung prüft Erreichbarkeit und Gültigkeit

2. **Einstellungen anpassen**: 
   - **Crawling-Tiefe**: Wie tief sollen Links verfolgt werden (Standard: 1)
   - **Seitenlimit**: Maximale Anzahl zu crawlender Seiten (Standard: 1)
   - **Erweiterte Optionen**: Chunk-Größe, Parallelisierung, Auto-Optimierung

3. **Wissensdatenbank benennen**: Eindeutigen Namen für die Datenbank vergeben

4. **Crawling starten**: Prozess wird mit Live-Progress-Tracking ausgeführt

### 2. 💬 Chat mit der Wissensdatenbank

1. **Datenbank auswählen**: Wähle eine der erstellten Wissensdatenbanken
2. **Fragen stellen**: Stelle natürlichsprachliche Fragen zu den Inhalten
3. **Antworten erhalten**: Erhalte kontextuelle Antworten basierend auf den gecrawlten Inhalten
4. **Chat exportieren**: Exportiere Gesprächsverläufe als Markdown-Datei

### 💡 Beispiel-Anwendungsfälle

- **Produktdokumentation**: Crawle Dokumentations-Websites für interne Wissensdatenbanken
- **Competitive Intelligence**: Analysiere Konkurrenz-Websites und stelle gezielte Fragen
- **Content Research**: Erstelle durchsuchbare Archive von Fachartikeln und Blogs
- **Support-Systeme**: Baue FAQ-Systeme basierend auf bestehenden Inhalten auf

## 🔧 Technische Details

### Backend-Architektur
- **Crawling Service**: Modal.com Serverless mit Crawl4AI und Playwright
- **Text Processing**: Intelligentes Chunking mit konfigurierbarer Größe (500-2500 Zeichen)
- **Embeddings**: Google Vertex AI `text-multilingual-embedding-002` für mehrsprachige Unterstützung
- **Vector Search**: ChromaDB mit Cosine Similarity und optionalem Reranking
- **LLM Integration**: Google Gemini 2.5 Flash für RAG-Antwortgenerierung

### Frontend-Technologien
- **UI Framework**: Streamlit mit modernem CSS-Styling
- **Session Management**: Persistente Zustände und Chat-Historie
- **Real-time Validation**: Debounced URL-Validierung mit Caching
- **Progress Tracking**: Live-Updates während Crawling-Prozess
- **Export-Funktionen**: Markdown-Export für Chat-Verläufe

### UX-Komponenten
- **URLValidator**: Real-time URL-Validierung mit visuellen Indikatoren
- **ProcessDisplay**: Detailliertes Progress-Tracking mit erweiterbaren Logs
- **SuccessAnimation**: Celebration-Animationen und Erfolgs-Metriken
- **AutoCompleteHandler**: Intelligente Namensvorschläge basierend auf Seitentiteln

### Sicherheit & Performance
- **Input-Validierung**: Umfassende URL- und Parameter-Validierung
- **Memory-Management**: Automatische Optimierung bei großen Datenmengen
- **Error-Handling**: Robuste Fehlerbehandlung mit benutzerfreundlichen Nachrichten
- **Caching**: Intelligentes Caching für URL-Validierung und Embeddings

## 📝 Lizenz

MIT License

## 🤝 Beitragen

Pull Requests sind willkommen! Für größere Änderungen öffne bitte zuerst ein Issue.

## 🆕 Neueste Updates

### Version 2.0 - UX Revolution
- **🎨 Komplett überarbeitete Benutzeroberfläche**: Minimalistisches, modernes Design
- **⚡ Real-time URL-Validierung**: Sofortige Überprüfung mit visuellen Indikatoren
- **🧠 Intelligente URL-Typ-Erkennung**: Automatische Optimierung basierend auf Website-Typ
- **🔄 Verbesserte Benutzerführung**: Schritt-für-Schritt Anleitung ohne Verwirrung
- **💬 Erweiterte Chat-Funktionen**: Export, Timestamps, bessere UX
- **🛡️ Robuste Fehlerbehandlung**: Benutzerfreundliche Fehlermeldungen und Tipps

### Version 1.5 - Performance & Stabilität
- **📈 Optimierte Crawling-Performance**: Bis zu 50% schnellere Verarbeitung
- **🔧 Modulare UX-Komponenten**: Wiederverwendbare UI-Bausteine
- **💾 Verbessertes Memory-Management**: Automatische Optimierung bei großen Datenmengen
- **🌐 Mehrsprachige Unterstützung**: Optimiert für deutsche und internationale Inhalte

## 🗺️ Roadmap

### Geplante Features
- [ ] **Batch-Upload**: Mehrere URLs gleichzeitig verarbeiten
- [ ] **Advanced Analytics**: Detaillierte Statistiken über Crawling-Ergebnisse
- [ ] **Custom Embeddings**: Support für weitere Embedding-Modelle
- [ ] **API-Zugang**: REST API für programmatischen Zugriff
- [ ] **Collaborative Features**: Team-basierte Wissensdatenbanken

## 📞 Support

Bei Fragen oder Problemen:
- 🐛 **Bugs**: Öffne ein [GitHub Issue](https://github.com/NicoDev01/Crawl_RAG_Agent_streamlit_demo/issues)
- 💡 **Feature Requests**: Diskutiere in [GitHub Discussions](https://github.com/NicoDev01/Crawl_RAG_Agent_streamlit_demo/discussions)
- 📧 **Direkte Anfragen**: Kontaktiere den Entwickler über GitHub