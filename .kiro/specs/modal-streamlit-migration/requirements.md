# Requirements Document

## Introduction

Dieses Projekt migriert das bestehende RAG-System von einer lokalen Architektur zu einer verteilten Cloud-Architektur. Das Crawling (mit Playwright/Crawl4AI) wird auf Modal.com ausgelagert, während die restlichen Komponenten (Embeddings, Vektordatenbank, RAG-Agent, UI) auf Streamlit Community Cloud laufen. Diese Aufteilung löst das Problem, dass Streamlit Community Cloud keine Headless-Browser unterstützt, während Modal.com optimal für rechenintensive, kurzlebige Tasks geeignet ist.

## Requirements

### Requirement 1: Modal.com Crawling Service

**User Story:** Als Entwickler möchte ich einen separaten Crawling-Service auf Modal.com haben, damit ich Playwright-basiertes Crawling ohne lokale Browser-Installation nutzen kann.

#### Acceptance Criteria

1. WHEN ein Crawling-Request an den Modal.com Service gesendet wird THEN soll der Service eine einzelne URL crawlen und strukturierte Daten zurückgeben
2. WHEN mehrere URLs gleichzeitig gecrawlt werden sollen THEN soll der Service Batch-Crawling mit konfigurierbarer Parallelität unterstützen
3. WHEN eine Website rekursiv gecrawlt werden soll THEN soll der Service interne Links folgen und die Tiefe begrenzen können
4. WHEN eine Sitemap-URL übergeben wird THEN soll der Service alle URLs aus der Sitemap extrahieren und crawlen
5. WHEN ein API-Request ohne gültigen Authorization-Header gesendet wird THEN soll der Service mit HTTP 401 antworten
6. WHEN ein Crawling-Fehler auftritt THEN soll der Service strukturierte Fehlermeldungen mit HTTP 500 zurückgeben
7. WHEN der Service deployed wird THEN soll er öffentlich über HTTPS-Endpunkte erreichbar sein

### Requirement 2: Crawler Client für Streamlit

**User Story:** Als Streamlit-Anwendung möchte ich mit dem Modal.com Crawling Service kommunizieren, damit ich Crawling-Funktionalität ohne lokale Browser nutzen kann.

#### Acceptance Criteria

1. WHEN die Streamlit-App eine URL crawlen möchte THEN soll der Crawler Client eine HTTP-Anfrage an den Modal.com Service senden
2. WHEN der Modal.com Service nicht erreichbar ist THEN soll der Client Retry-Mechanismen mit exponential backoff verwenden
3. WHEN API-Credentials konfiguriert sind THEN soll der Client diese automatisch in Authorization-Headern verwenden
4. WHEN verschiedene Crawling-Modi benötigt werden THEN soll der Client alle Service-Endpunkte (single, batch, recursive, sitemap) unterstützen
5. WHEN ein API-Fehler auftritt THEN soll der Client aussagekräftige Fehlermeldungen weiterleiten

### Requirement 3: Angepasste Ingestion Pipeline

**User Story:** Als Benutzer möchte ich weiterhin Webseiten in meine Wissensdatenbank aufnehmen können, auch wenn das Crawling jetzt über einen externen Service läuft.

#### Acceptance Criteria

1. WHEN eine URL zur Ingestion eingegeben wird THEN soll die Pipeline den Modal.com Service für das Crawling verwenden
2. WHEN gecrawlte Inhalte zurückkommen THEN soll die Pipeline diese wie bisher in Chunks aufteilen
3. WHEN Chunks erstellt wurden THEN soll die Pipeline Vertex AI Embeddings generieren
4. WHEN Embeddings erstellt wurden THEN soll die Pipeline diese in ChromaDB speichern
5. WHEN die Ingestion fehlschlägt THEN soll die Pipeline aussagekräftige Fehlermeldungen anzeigen
6. WHEN verschiedene URL-Typen (einzeln, Sitemap, Textdatei) eingegeben werden THEN soll die Pipeline den entsprechenden Crawling-Modus wählen

### Requirement 4: Streamlit Community Cloud Deployment

**User Story:** Als Benutzer möchte ich die RAG-Anwendung über eine öffentlich zugängliche Web-URL nutzen können, ohne lokale Installation.

#### Acceptance Criteria

1. WHEN die Streamlit-App deployed wird THEN soll sie über eine öffentliche URL erreichbar sein
2. WHEN Google Cloud Credentials benötigt werden THEN sollen diese sicher über Streamlit Secrets verwaltet werden
3. WHEN Modal.com API-Credentials benötigt werden THEN sollen diese über Streamlit Secrets konfiguriert werden
4. WHEN die App startet THEN soll sie automatisch alle benötigten Services (Vertex AI, ChromaDB) initialisieren
5. WHEN ein Benutzer die App nutzt THEN soll die Funktionalität identisch zur lokalen Version sein
6. WHEN ChromaDB auf Streamlit Community Cloud läuft THEN soll das SQLite3-Kompatibilitätsproblem durch pysqlite3-binary gelöst werden
7. WHEN die App startet THEN soll der SQLite3-Hack vor allen anderen Imports ausgeführt werden
8. WHEN ChromaDB initialisiert wird THEN soll es im flüchtigen Speicher des Containers laufen
9. WHEN die App neustartet THEN sollen bestehende Collections automatisch wiederhergestellt werden

### Requirement 5: Sichere API-Kommunikation

**User Story:** Als Systemadministrator möchte ich sicherstellen, dass die Kommunikation zwischen Streamlit und Modal.com Service sicher und authentifiziert ist.

#### Acceptance Criteria

1. WHEN API-Requests an Modal.com gesendet werden THEN sollen diese einen Bearer Token im Authorization-Header enthalten
2. WHEN API-Keys konfiguriert werden THEN sollen diese als Secrets und nicht im Code gespeichert werden
3. WHEN ungültige API-Keys verwendet werden THEN soll der Service mit HTTP 401 antworten
4. WHEN API-Keys in Modal.com konfiguriert werden THEN sollen diese über das Modal Secrets System verwaltet werden
5. WHEN API-Keys in Streamlit konfiguriert werden THEN sollen diese über Streamlit Secrets verwaltet werden

### Requirement 6: Fehlerbehandlung und Monitoring

**User Story:** Als Entwickler möchte ich aussagekräftige Fehlermeldungen und Logs haben, damit ich Probleme schnell identifizieren und beheben kann.

#### Acceptance Criteria

1. WHEN ein Crawling-Fehler auf Modal.com auftritt THEN sollen detaillierte Logs in der Modal.com Console erscheinen
2. WHEN ein API-Fehler in der Streamlit-App auftritt THEN soll eine benutzerfreundliche Fehlermeldung angezeigt werden
3. WHEN Retry-Mechanismen aktiviert werden THEN sollen diese in den Logs dokumentiert werden
4. WHEN die Ingestion läuft THEN soll der Fortschritt in der Streamlit-UI angezeigt werden
5. WHEN Services nicht erreichbar sind THEN sollen Timeout-Fehler klar kommuniziert werden

### Requirement 7: ChromaDB Cloud-Kompatibilität

**User Story:** Als Entwickler möchte ich ChromaDB auf Streamlit Community Cloud zum Laufen bringen, trotz der SQLite-Inkompatibilität und Speicherbeschränkungen.

#### Acceptance Criteria

1. WHEN die App deployed wird THEN soll eine packages.txt Datei libsqlite3-dev installieren
2. WHEN Python startet THEN soll pysqlite3-binary als sqlite3 Ersatz geladen werden
3. WHEN ChromaDB initialisiert wird THEN soll es einen In-Memory Client verwenden
4. WHEN Collections erstellt werden THEN sollen diese mit @st.cache_resource gecacht werden
5. WHEN die App neustartet THEN sollen Collections automatisch neu erstellt werden
6. WHEN der Speicherverbrauch zu hoch wird THEN soll die App graceful degradieren
7. WHEN Collections zu groß werden THEN sollen diese in kleinere Batches aufgeteilt werden

### Requirement 8: Backward Compatibility

**User Story:** Als bestehender Benutzer möchte ich meine vorhandenen Wissensdatenbanken weiterhin nutzen können, auch nach der Migration.

#### Acceptance Criteria

1. WHEN die neue Version deployed wird THEN sollen bestehende ChromaDB Collections konzeptuell weiterhin funktionieren
2. WHEN der RAG-Agent Fragen beantwortet THEN soll die Antwortqualität identisch zur lokalen Version sein
3. WHEN die Streamlit-UI geladen wird THEN sollen alle bestehenden Features verfügbar sein
4. WHEN Embeddings generiert werden THEN sollen diese kompatibel zu den bestehenden Embeddings sein
5. WHEN Collections in der Cloud neu erstellt werden THEN sollen diese die gleichen Namen und Strukturen haben