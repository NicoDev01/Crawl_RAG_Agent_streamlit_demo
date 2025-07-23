# Modal.com Crawl4AI Service

Serverless Web-Crawling Service basierend auf Crawl4AI und Playwright, gehostet auf Modal.com.

## Setup

### 1. Modal.com Account und CLI

```bash
# Modal CLI installieren
pip install modal

# Authentifizierung (öffnet Browser)
python3 -m modal setup
```

### 2. API-Schlüssel konfigurieren

1. Gehe zum [Modal Dashboard](https://modal.com/secrets)
2. Erstelle ein neues Secret namens `crawl4ai-api-key`
3. Füge einen Eintrag hinzu:
   - Key: `API_KEY`
   - Value: `dein-sicherer-api-schluessel` (z.B. generiert mit `openssl rand -hex 32`)

### 3. Service deployen

```bash
cd modal_crawler
modal deploy crawler_service.py
```

Nach dem Deployment erhältst du URLs wie:
- `https://your-app-name--crawl-single.modal.run`
- `https://your-app-name--health-check.modal.run`

## API Endpunkte

### Health Check
```bash
GET https://your-app-name--health-check.modal.run
```

### Single URL Crawling
```bash
POST https://your-app-name--crawl-single.modal.run
Content-Type: application/json
Authorization: Bearer your-api-key

{
    "url": "https://example.com",
    "cache_mode": "BYPASS"
}
```

**Response:**
```json
{
    "success": true,
    "url": "https://example.com",
    "markdown": "# Example Domain\n\nThis domain is for use in illustrative examples...",
    "links": {
        "internal": [],
        "external": [...]
    }
}
```

## Testen

1. Aktualisiere `test_service.py` mit deinen URLs und API-Key
2. Führe Tests aus:
```bash
python test_service.py
```

## Entwicklung

### Lokaler Test
```bash
modal run crawler_service.py
```

### Logs anzeigen
```bash
modal logs crawl4ai-service
```

### Service stoppen
```bash
modal app stop crawl4ai-service
```

## Nächste Schritte

Nach erfolgreichem Deployment des Single URL Endpunkts:
1. Batch Crawling Endpunkt hinzufügen
2. Recursive Crawling Endpunkt implementieren
3. Sitemap Crawling Endpunkt erstellen
4. Streamlit Client für API-Kommunikation entwickeln