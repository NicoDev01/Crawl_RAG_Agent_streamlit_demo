# Requirements Document

## Introduction

CraCha funktioniert lokal perfekt, aber auf Streamlit Community Cloud treten Memory- und Timeout-Probleme auf. Das System muss für die Cloud-Umgebung optimiert werden, ohne die Kernfunktionalität zu beeinträchtigen.

## Requirements

### Requirement 1: Memory-Optimierung für Streamlit Cloud

**User Story:** Als Streamlit Cloud User möchte ich, dass CraCha auch bei großen Datasets (3000+ Chunks) stabil läuft, damit ich das Tool produktiv nutzen kann.

#### Acceptance Criteria

1. WHEN ein großes Dataset (>3000 Chunks) verarbeitet wird THEN soll das System unter 1GB Memory bleiben
2. WHEN ChromaDB Standard-Embeddings verwendet werden THEN soll das ONNX-Modell effizient geladen werden
3. WHEN die Verarbeitung länger als 10 Minuten dauert THEN soll der Health-Check nicht timeout
4. WHEN Memory-Limits erreicht werden THEN soll das System automatisch die Batch-Größe reduzieren

### Requirement 2: Streamlit Cloud Health-Check Kompatibilität

**User Story:** Als Streamlit Cloud Nutzer möchte ich, dass die App nicht nach 19 Minuten abstürzt, damit auch große Crawls erfolgreich abgeschlossen werden.

#### Acceptance Criteria

1. WHEN ein Crawl länger als 15 Minuten dauert THEN soll das System regelmäßig Health-Signals senden
2. WHEN die Verarbeitung läuft THEN soll der UI-Thread responsive bleiben
3. WHEN ein Timeout droht THEN soll das System den Prozess in kleinere Teile aufteilen

### Requirement 3: Intelligente Dataset-Reduktion

**User Story:** Als User möchte ich bei großen Datasets automatische Optimierungen, damit das System auch auf begrenzten Ressourcen funktioniert.

#### Acceptance Criteria

1. WHEN ein Dataset >5000 Chunks hat THEN soll automatisch auf die wichtigsten Chunks reduziert werden
2. WHEN Memory-Probleme erkannt werden THEN soll die Chunk-Größe automatisch angepasst werden
3. WHEN die Cloud-Umgebung erkannt wird THEN sollen konservative Limits verwendet werden

### Requirement 4: Robuste Error-Handling für Cloud-Deployment

**User Story:** Als Entwickler möchte ich aussagekräftige Fehlermeldungen bei Cloud-spezifischen Problemen, damit ich Probleme schnell identifizieren kann.

#### Acceptance Criteria

1. WHEN Memory-Limits überschritten werden THEN soll eine klare Fehlermeldung mit Lösungsvorschlägen angezeigt werden
2. WHEN Timeouts auftreten THEN soll der User über alternative Strategien informiert werden
3. WHEN ChromaDB-Probleme auftreten THEN sollen Fallback-Optionen angeboten werden

### Requirement 5: Progressive Processing für große Datasets

**User Story:** Als User möchte ich auch bei sehr großen Websites (100+ Seiten) eine funktionierende Verarbeitung, damit ich nicht auf kleinere Datasets beschränkt bin.

#### Acceptance Criteria

1. WHEN ein großes Dataset verarbeitet wird THEN soll es in mehrere Sessions aufgeteilt werden können
2. WHEN ein Prozess unterbrochen wird THEN sollen bereits verarbeitete Daten erhalten bleiben
3. WHEN die Verarbeitung fortgesetzt wird THEN soll nahtlos an der letzten Position weitergemacht werden