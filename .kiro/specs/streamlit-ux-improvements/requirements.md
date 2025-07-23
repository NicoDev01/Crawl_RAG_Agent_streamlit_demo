# Requirements Document

## Introduction

Diese Spezifikation definiert UX-Verbesserungen für die Streamlit RAG Knowledge Assistant App, um die Benutzerfreundlichkeit und das Feedback während des Crawling-Prozesses erheblich zu verbessern. Die Verbesserungen umfassen eine detaillierte Prozessanzeige, URL-Validierung, Auto-Complete-Funktionen, Website-Erkennung und visuelle Erfolgs-Feedbacks.

## Requirements

### Requirement 1: Erweiterte Prozessanzeige

**User Story:** Als Benutzer möchte ich detailliert sehen was während des Crawling-Prozesses passiert, damit ich den Fortschritt verfolgen und bei Problemen eingreifen kann.

#### Acceptance Criteria

1. WHEN der Crawling-Prozess startet THEN soll eine detaillierte Fortschrittsanzeige mit aktueller URL angezeigt werden
2. WHEN eine URL gecrawlt wird THEN soll die aktuelle URL in der Prozessanzeige sichtbar sein
3. WHEN der Prozess läuft THEN soll ein expandierbares Crawling-Log alle gefundenen/gecrawlten URLs anzeigen
4. WHEN Fehler auftreten THEN sollen diese im Log mit Details angezeigt werden
5. WHEN der Prozess abgeschlossen ist THEN soll eine Zusammenfassung mit Statistiken angezeigt werden

### Requirement 2: Erfolgsanimation und visuelles Feedback

**User Story:** Als Benutzer möchte ich eine ansprechende Bestätigung wenn das Crawling erfolgreich war, damit ich positive Verstärkung für die erfolgreiche Aktion erhalte.

#### Acceptance Criteria

1. WHEN das Crawling erfolgreich abgeschlossen ist THEN soll eine Konfetti-Animation oder ähnliche Celebration angezeigt werden
2. WHEN die Animation läuft THEN soll sie nicht länger als 3 Sekunden dauern
3. WHEN die Wissensdatenbank erstellt wurde THEN sollen die Erfolgsmetriken prominent und visuell ansprechend dargestellt werden
4. WHEN der Erfolg angezeigt wird THEN soll ein Call-to-Action zum Chat-Tab sichtbar sein

### Requirement 3: URL-Validierung in Echtzeit

**User Story:** Als Benutzer möchte ich sofort sehen ob meine eingegebene URL gültig und erreichbar ist, damit ich Fehler vor dem Crawling-Start erkennen kann.

#### Acceptance Criteria

1. WHEN eine URL eingegeben wird THEN soll in Echtzeit validiert werden ob die URL syntaktisch korrekt ist
2. WHEN eine gültige URL eingegeben wird THEN soll ein grüner Indikator angezeigt werden
3. WHEN eine ungültige URL eingegeben wird THEN soll ein roter Indikator mit Fehlermeldung angezeigt werden
4. WHEN eine URL erreichbar ist THEN soll zusätzlich ein Erreichbarkeits-Check durchgeführt werden
5. WHEN eine URL nicht erreichbar ist THEN soll eine entsprechende Warnung angezeigt werden

### Requirement 4: Auto-Complete für Datenbankname

**User Story:** Als Benutzer möchte ich Vorschläge für den Datenbanknamen basierend auf der URL erhalten, damit ich nicht selbst einen Namen erfinden muss.

#### Acceptance Criteria

1. WHEN eine gültige URL eingegeben wird THEN soll automatisch ein passender Datenbankname vorgeschlagen werden
2. WHEN die URL eine bekannte Domain enthält THEN soll der Vorschlag auf dem Domain-Namen basieren
3. WHEN der Benutzer den Vorschlag übernimmt THEN soll das Namensfeld automatisch ausgefüllt werden
4. WHEN mehrere Vorschläge möglich sind THEN sollen diese in einem Dropdown angezeigt werden
5. WHEN der Benutzer einen eigenen Namen eingibt THEN sollen die Vorschläge ausgeblendet werden

### Requirement 5: Intelligente Website-Erkennung

**User Story:** Als Benutzer möchte ich dass die App automatisch passende Einstellungen für verschiedene Website-Typen vorschlägt, damit ich nicht alle Parameter manuell konfigurieren muss.

#### Acceptance Criteria

1. WHEN eine URL eingegeben wird THEN soll der Website-Typ automatisch erkannt werden (Dokumentation, Blog, News, etc.)
2. WHEN ein Website-Typ erkannt wird THEN sollen passende Default-Einstellungen vorgeschlagen werden
3. WHEN es sich um eine Dokumentations-Website handelt THEN sollen höhere Crawling-Tiefe und Seitenzahl vorgeschlagen werden
4. WHEN es sich um einen Blog handelt THEN sollen moderate Einstellungen vorgeschlagen werden
5. WHEN der Website-Typ unbekannt ist THEN sollen konservative Standard-Einstellungen verwendet werden
6. WHEN Einstellungen vorgeschlagen werden THEN soll der Benutzer diese überschreiben können

### Requirement 6: Sitemap Auto-Detection

**User Story:** Als Benutzer möchte ich dass die App automatisch nach einer Sitemap sucht, damit ich nicht manuell nach sitemap.xml URLs suchen muss.

#### Acceptance Criteria

1. WHEN eine Website-URL eingegeben wird THEN soll automatisch nach einer Sitemap gesucht werden
2. WHEN eine Sitemap gefunden wird THEN soll ein Button "Sitemap verwenden" angezeigt werden
3. WHEN der Sitemap-Button geklickt wird THEN soll automatisch auf "Sitemap" Crawling-Typ gewechselt werden
4. WHEN keine Sitemap gefunden wird THEN soll keine Sitemap-Option angezeigt werden
5. WHEN die Sitemap-Suche fehlschlägt THEN soll dies nicht den normalen Workflow blockieren

### Requirement 7: Verbesserte Fehlerbehandlung und Benutzerführung

**User Story:** Als Benutzer möchte ich klare Fehlermeldungen und Lösungsvorschläge erhalten, damit ich Probleme selbst beheben kann.

#### Acceptance Criteria

1. WHEN ein Fehler auftritt THEN soll eine benutzerfreundliche Fehlermeldung angezeigt werden
2. WHEN möglich THEN sollen konkrete Lösungsvorschläge angeboten werden
3. WHEN ein Netzwerkfehler auftritt THEN soll ein Retry-Button angezeigt werden
4. WHEN die Eingaben unvollständig sind THEN sollen die fehlenden Felder hervorgehoben werden
5. WHEN ein kritischer Fehler auftritt THEN soll der Benutzer die Möglichkeit haben den Support zu kontaktieren