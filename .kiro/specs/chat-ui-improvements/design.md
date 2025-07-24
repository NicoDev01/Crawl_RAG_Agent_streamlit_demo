# Chat UI & Performance Improvements - Design

## Overview

Diese Design-Spezifikation beschreibt die technische Umsetzung der Chat-Interface-Verbesserungen und Performance-Optimierungen.

## Architecture

### Component Structure
```
streamlit_app.py
├── render_improved_chat_interface()
│   ├── ChatProgressIndicator (neu)
│   ├── StreamingResponseHandler (neu)
│   └── AsyncRAGProcessor (neu)
├── rag_agent.py
│   ├── async_retrieve_context()
│   ├── parallel_hyde_embedding()
│   └── confidence_calculator()
└── ui_components.py (neu)
    ├── ProgressSteps
    ├── StreamingText
    └── ConfidenceIndicator
```

## Components and Interfaces

### 1. ChatProgressIndicator
**Zweck:** Zeigt detaillierte Fortschrittsinformationen während der RAG-Verarbeitung

**Interface:**
```python
class ProgressStep:
    name: str
    status: Literal["pending", "running", "completed", "error"]
    message: str
    duration: Optional[float]

class ChatProgressIndicator:
    def __init__(self, steps: List[str])
    def update_step(self, step_name: str, status: str, message: str)
    def render(self) -> None
```

### 2. StreamingResponseHandler
**Zweck:** Implementiert Echtzeit-Streaming von Gemini-Antworten

**Interface:**
```python
class StreamingResponseHandler:
    def __init__(self, placeholder: st.empty)
    async def stream_response(self, response_generator) -> str
    def _render_partial_text(self, text: str, show_cursor: bool)
```

### 3. AsyncRAGProcessor
**Zweck:** Koordiniert asynchrone RAG-Verarbeitung

**Interface:**
```python
class AsyncRAGProcessor:
    async def process_question(self, question: str, deps: RAGDeps) -> str
    async def _parallel_hyde_embedding(self, question: str) -> Tuple[str, List[float]]
    async def _multi_query_retrieval(self, question: str) -> List[Dict]
    def _calculate_confidence(self, reranker_scores: List[float]) -> float
```

## Data Models

### ProgressState
```python
@dataclass
class ProgressState:
    current_step: int
    total_steps: int
    steps: List[ProgressStep]
    start_time: datetime
    estimated_completion: Optional[datetime]
```

### ConfidenceScore
```python
@dataclass
class ConfidenceScore:
    score: float  # 0.0 - 1.0
    level: Literal["high", "medium", "low"]
    explanation: str
    chunk_scores: List[float]
```

### StreamingConfig
```python
@dataclass
class StreamingConfig:
    chunk_size: int = 10  # Zeichen pro Chunk
    delay_ms: int = 50    # Verzögerung zwischen Chunks
    show_cursor: bool = True
    cursor_char: str = "▌"
```

## Error Handling

### Streaming Fallback
- Bei Streaming-Fehlern: Fallback auf normale Anzeige
- Bei Timeout: Zeige Teilantwort mit Warnung
- Bei API-Fehlern: Zeige Fehlermeldung mit Retry-Option

### Async Error Recovery
- Parallele Prozesse: Ein Fehler stoppt nicht andere Prozesse
- Timeout-Handling: Automatischer Fallback nach 30 Sekunden
- Graceful Degradation: System funktioniert auch ohne einzelne Features

## Testing Strategy

### Unit Tests
- `test_progress_indicator.py`: Fortschrittsanzeige-Logik
- `test_streaming_handler.py`: Streaming-Funktionalität
- `test_async_processor.py`: Asynchrone Verarbeitung

### Integration Tests
- `test_chat_flow.py`: Kompletter Chat-Ablauf
- `test_error_scenarios.py`: Fehlerbehandlung
- `test_performance.py`: Performance-Messungen

### UI Tests
- `test_visual_components.py`: CSS und Layout
- `test_responsiveness.py`: Mobile/Desktop Darstellung
- `test_accessibility.py`: Barrierefreiheit

## Performance Considerations

### Async Optimizations
- HyDE und Embedding parallel ausführen
- Multi-Query Retrieval mit asyncio.gather()
- Non-blocking UI Updates

### Caching Strategy
- Cache für häufige Fragen (TTL: 1 Stunde)
- Embedding-Cache für wiederholte Queries
- Progress-State in Session State

### Memory Management
- Streaming reduziert Memory-Peak
- Async Tasks cleanup nach Completion
- Garbage Collection für große Responses