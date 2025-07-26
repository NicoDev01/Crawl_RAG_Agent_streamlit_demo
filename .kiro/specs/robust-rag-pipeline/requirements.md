# Requirements Document

## Introduction

Das aktuelle RAG-System zeigt themenspezifische Recall-Verluste, die durch suboptimale Pipeline-Architektur entstehen. Diese Spec implementiert ein robustes, themen-agnostisches RAG-System basierend auf bewährten Best Practices für Crawling, Embedding, Retrieval und Re-Ranking.

## Requirements

### Requirement 1: Robuste Crawling & Pre-Processing Pipeline

**User Story:** Als Entwickler möchte ich eine konsistente Crawling-Pipeline, die semantischen Kontext erhält und für alle Website-Typen funktioniert, damit keine wichtigen Informationen verloren gehen.

#### Acceptance Criteria

1. WHEN JavaScript-basierte Seiten gecrawlt werden THEN soll Playwright mit DOMContentLoaded-Parsing verwendet werden
2. WHEN Content in Chunks aufgeteilt wird THEN soll die Aufteilung an Überschrift-Boundaries (h1-h6, aria-level) erfolgen
3. WHEN Chunks erstellt werden THEN sollen sie 30-256 Tokens enthalten (statt ≥10 Zeichen)
4. WHEN Dokumente verarbeitet werden THEN sollen Metadaten (source_url, heading_path, crawl_timestamp, lang) gespeichert werden
5. WHEN Chunks generiert werden THEN soll jeder Chunk Überschrift + 1-n Absätze enthalten für semantischen Kontext

### Requirement 2: Konsistente Embedding-Strategie

**User Story:** Als System möchte ich einheitliche Embeddings für alle Dokumente und Queries verwenden, damit die Vektorsuche zuverlässig funktioniert.

#### Acceptance Criteria

1. WHEN Embeddings generiert werden THEN soll dasselbe Modell (gemini-embedding-001) mit gleicher Dimension (768) verwendet werden
2. WHEN Dokument-Embeddings erstellt werden THEN soll task_type="RETRIEVAL_DOCUMENT" verwendet werden
3. WHEN Query-Embeddings erstellt werden THEN soll task_type="RETRIEVAL_QUERY" verwendet werden
4. WHEN Embeddings normiert werden THEN soll L2-Normalisierung angewendet werden
5. WHEN ChromaDB Collection erstellt wird THEN sollen Embedding-Metadaten (model, dimension) persistiert werden

### Requirement 3: High-Recall Retrieval Layer

**User Story:** Als RAG-System möchte ich einen hohen Recall bei der Kandidatensuche erreichen, damit relevante Dokumente nicht vorzeitig gefiltert werden.

#### Acceptance Criteria

1. WHEN Retrieval durchgeführt wird THEN soll Hybrid Search (BM25 + Embeddings) implementiert werden
2. WHEN Queries verarbeitet werden THEN sollen Multi-Query-Variationen (Synonyme, Frage↔Aussage) generiert werden
3. WHEN Kandidaten abgerufen werden THEN soll k ≈ max(200, n_final * 10) verwendet werden
4. WHEN Filtering angewendet wird THEN sollen nur Duplikate entfernt werden, keine Score-Schwellen
5. WHEN keine Ergebnisse gefunden werden THEN soll Fallback auf Cosine-Similarity-Sort erfolgen

### Requirement 4: Domänen-agnostischer Re-Ranking Layer

**User Story:** Als System möchte ich einen robusten Re-Ranking-Mechanismus, der unabhängig vom Fachgebiet funktioniert, damit die besten Ergebnisse priorisiert werden.

#### Acceptance Criteria

1. WHEN Re-Ranking durchgeführt wird THEN soll Vertex AI Ranking API als primäre Methode verwendet werden
2. WHEN Vertex AI nicht verfügbar ist THEN soll Cross-Encoder-Fallback implementiert werden
3. WHEN beide Methoden fehlschlagen THEN soll Cosine-Similarity-Fallback verwendet werden
4. WHEN Re-Ranking-Scores berechnet werden THEN sollen sie domänen-agnostisch sein (keine Token-Overlap-Heuristik)
5. WHEN Top-N Ergebnisse ausgewählt werden THEN soll die Auswahl nach Re-Ranking-Score erfolgen

### Requirement 5: Fail-Open Strategie

**User Story:** Als Benutzer möchte ich immer eine Antwort erhalten, auch wenn nicht alle relevanten Chunks gefunden werden, damit das System robust und benutzerfreundlich bleibt.

#### Acceptance Criteria

1. WHEN weniger als gewünschte Chunks gefunden werden THEN soll das System trotzdem eine Antwort generieren
2. WHEN keine spezifischen Informationen verfügbar sind THEN soll dies transparent kommuniziert werden
3. WHEN Fallback-Chunks benötigt werden THEN sollen sie nach Cosine-Score sortiert hinzugefügt werden
4. WHEN Antworten generiert werden THEN soll nie "gibt es nicht" gesagt werden, wenn Unsicherheit besteht
5. WHEN unvollständige Informationen vorliegen THEN sollen Handlungsempfehlungen gegeben werden

### Requirement 6: Monitoring & Evaluation System

**User Story:** Als Entwickler möchte ich die Pipeline-Performance überwachen können, damit ich Probleme frühzeitig erkenne und behebe.

#### Acceptance Criteria

1. WHEN das System läuft THEN soll Answer Empty-Rate (< 2%) gemessen werden
2. WHEN Evaluation durchgeführt wird THEN soll Recall@k für verschiedene Themen berechnet werden
3. WHEN Chunks verworfen werden THEN soll geloggt werden, in welchem Pipeline-Step dies geschah
4. WHEN Performance-Metriken gesammelt werden THEN sollen sie themen-unabhängig sein
5. WHEN Goldsets verwendet werden THEN sollen sie automatisiert via LangChain-Bench evaluiert werden

### Requirement 7: Intelligente Stop-Word-Behandlung

**User Story:** Als System möchte ich Stop-Words intelligent handhaben, damit sie bei deutschen Overlap-Scores helfen, aber englische oder fachspezifische Abfragen nicht beeinträchtigen.

#### Acceptance Criteria

1. WHEN Sprache erkannt wird THEN soll das entsprechende Stop-Word-Set verwendet werden
2. WHEN Embedding-Flow durchgeführt wird THEN sollen Stop-Words optional sein
3. WHEN Keyword-Heuristiken angewendet werden THEN sollen Stop-Words verwendet werden
4. WHEN mehrsprachige Inhalte verarbeitet werden THEN soll Spracherkennung (langdetect) implementiert werden
5. WHEN fachspezifische Begriffe vorkommen THEN sollen sie nicht als Stop-Words behandelt werden

### Requirement 8: Strukturierte Pipeline-Architektur

**User Story:** Als Entwickler möchte ich eine klar getrennte Pipeline-Architektur, damit jede Komponente unabhängig optimiert und getestet werden kann.

#### Acceptance Criteria

1. WHEN Pipeline-Komponenten implementiert werden THEN sollen sie klar getrennte Verantwortlichkeiten haben
2. WHEN Candidate Generation durchgeführt wird THEN soll sie von Re-Ranking getrennt sein
3. WHEN Filtering angewendet wird THEN soll es erst am Ende der Pipeline erfolgen
4. WHEN Komponenten kommunizieren THEN sollen strukturierte Datenmodelle verwendet werden
5. WHEN Tests geschrieben werden THEN soll jede Komponente isoliert testbar sein

### Requirement 9: Adaptive Chunk-Strategien

**User Story:** Als System möchte ich adaptive Chunking-Strategien verwenden, damit verschiedene Content-Typen optimal verarbeitet werden.

#### Acceptance Criteria

1. WHEN strukturierter Content (Dokumentation) verarbeitet wird THEN soll Heading-basiertes Chunking verwendet werden
2. WHEN unstrukturierter Content verarbeitet wird THEN soll semantisches Chunking angewendet werden
3. WHEN Code-Blöcke erkannt werden THEN sollen sie als zusammenhängende Einheiten behandelt werden
4. WHEN Listen oder Tabellen verarbeitet werden THEN soll ihre Struktur erhalten bleiben
5. WHEN Chunk-Größen bestimmt werden THEN sollen sie content-type-spezifisch optimiert werden

### Requirement 10: Erweiterte Query-Verarbeitung

**User Story:** Als Benutzer möchte ich, dass meine Fragen intelligent verarbeitet werden, damit auch komplexe oder mehrdeutige Anfragen korrekt beantwortet werden.

#### Acceptance Criteria

1. WHEN Fragen analysiert werden THEN soll Intent-Erkennung implementiert werden
2. WHEN Query-Variationen generiert werden THEN sollen sie semantisch äquivalent aber lexikalisch unterschiedlich sein
3. WHEN Synonyme verwendet werden THEN sollen sie domänen-spezifisch sein
4. WHEN Frage-zu-Aussage-Transformation durchgeführt wird THEN soll der semantische Gehalt erhalten bleiben
5. WHEN mehrere Query-Strategien verwendet werden THEN sollen sie parallel ausgeführt werden