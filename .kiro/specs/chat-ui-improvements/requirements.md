# Chat UI & Performance Improvements

## Introduction

Diese Spec definiert Verbesserungen für die Chat-Benutzeroberfläche und Performance-Optimierungen des RAG-Systems. Ziel ist es, eine professionellere UI mit Echtzeit-Feedback und besserer Responsiveness zu schaffen.

## Requirements

### Requirement 1: Visuelles Chat-Interface verbessern

**User Story:** Als Benutzer möchte ich ein professionell aussehendes Chat-Interface ohne störende weiße Ränder, damit die Nutzung angenehmer wird.

#### Acceptance Criteria

1. WHEN der Benutzer das Chat-Interface öffnet THEN soll das Eingabefeld ohne weiße Box-Ränder dargestellt werden
2. WHEN der Benutzer eine Nachricht eingibt THEN soll das Design konsistent und modern aussehen
3. WHEN der Chat angezeigt wird THEN sollen die Nachrichten-Container visuell ansprechend gestaltet sein
4. WHEN das Interface geladen wird THEN soll es responsive und auf verschiedenen Bildschirmgrößen gut aussehen

### Requirement 2: Echtzeit-Prozess-Feedback implementieren

**User Story:** Als Benutzer möchte ich genau sehen, was im Hintergrund passiert (HyDE, Retrieval, Reranking), damit ich verstehe, warum die Antwort Zeit braucht.

#### Acceptance Criteria

1. WHEN eine Frage gestellt wird THEN soll ein detaillierter Fortschrittsindikator angezeigt werden
2. WHEN HyDE ausgeführt wird THEN soll "🧠 Generiere hypothetische Antwort..." angezeigt werden
3. WHEN ChromaDB durchsucht wird THEN soll "🔍 Durchsuche Wissensdatenbank (50 Kandidaten)..." angezeigt werden
4. WHEN Vertex AI Reranking läuft THEN soll "⚡ Sortiere nach Relevanz..." angezeigt werden
5. WHEN Gemini die Antwort generiert THEN soll "✍️ Formuliere Antwort..." angezeigt werden
6. WHEN ein Schritt abgeschlossen ist THEN soll ein Häkchen (✅) angezeigt werden

### Requirement 3: Streaming Responses implementieren

**User Story:** Als Benutzer möchte ich die Antwort sehen, während sie generiert wird, damit ich nicht auf die komplette Antwort warten muss.

#### Acceptance Criteria

1. WHEN Gemini eine Antwort generiert THEN soll der Text Wort für Wort erscheinen
2. WHEN der Text gestreamt wird THEN soll ein blinkender Cursor (▌) das Ende markieren
3. WHEN das Streaming abgeschlossen ist THEN soll der Cursor verschwinden
4. WHEN ein Fehler beim Streaming auftritt THEN soll die komplette Antwort auf einmal angezeigt werden

### Requirement 4: Async Processing für bessere Performance

**User Story:** Als Benutzer möchte ich, dass das Interface während der Verarbeitung responsiv bleibt, damit ich andere Aktionen ausführen kann.

#### Acceptance Criteria

1. WHEN eine RAG-Anfrage läuft THEN soll das Interface nicht blockiert werden
2. WHEN HyDE und Embedding parallel ausgeführt werden THEN soll die Gesamtzeit reduziert werden
3. WHEN mehrere Prozesse laufen THEN sollen sie asynchron abgearbeitet werden
4. WHEN ein Prozess fehlschlägt THEN sollen andere Prozesse weiterlaufen können

### Requirement 5: Multi-Query Retrieval (Optional)

**User Story:** Als Benutzer möchte ich bessere Suchergebnisse durch mehrere Suchvarianten, damit relevante Informationen nicht übersehen werden.

#### Acceptance Criteria

1. WHEN eine Frage gestellt wird THEN sollen 3-4 Varianten der Frage generiert werden
2. WHEN die Varianten erstellt sind THEN sollen sie parallel durchsucht werden
3. WHEN alle Suchen abgeschlossen sind THEN sollen die Ergebnisse intelligent kombiniert werden
4. WHEN die finale Antwort generiert wird THEN soll sie auf allen gefundenen Informationen basieren

### Requirement 6: Confidence Scoring anzeigen

**User Story:** Als Benutzer möchte ich wissen, wie sicher sich das System bei der Antwort ist, damit ich die Verlässlichkeit einschätzen kann.

#### Acceptance Criteria

1. WHEN eine Antwort generiert wird THEN soll ein Confidence Score (0-100%) berechnet werden
2. WHEN der Score hoch ist (>80%) THEN soll ein grünes Vertrauens-Icon angezeigt werden
3. WHEN der Score mittel ist (50-80%) THEN soll ein gelbes Vorsichts-Icon angezeigt werden
4. WHEN der Score niedrig ist (<50%) THEN soll ein rotes Warnung-Icon angezeigt werden
5. WHEN der Score angezeigt wird THEN soll eine Erklärung verfügbar sein