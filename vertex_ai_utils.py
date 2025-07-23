import time
import os
from typing import List, Optional
from google.cloud import aiplatform
from google.auth.exceptions import DefaultCredentialsError
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

# Konfiguration für Vertex AI
# Diese sollten idealerweise über Umgebungsvariablen oder eine Konfigurationsdatei geladen werden
# Für das Testing gehen wir davon aus, dass GOOGLE_APPLICATION_CREDENTIALS gesetzt ist.

# Globale Initialisierung des Vertex AI Clients vermeiden, stattdessen bei Bedarf erstellen oder übergeben.

def init_vertex_ai(project_id: Optional[str] = None, location: Optional[str] = None):
    """Initializes Vertex AI client with specified project and location."""
    final_project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
    final_location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    if not final_project_id:
        raise ValueError("Google Cloud Project ID not set. Please set GOOGLE_CLOUD_PROJECT environment variable or pass as argument.")

    try:
        aiplatform.init(project=final_project_id, location=final_location)
        print(f"Vertex AI initialized for project '{final_project_id}' in location '{final_location}'.")
    except DefaultCredentialsError:
        print("--------------------------------------------------------------------------------------")
        print("Fehler bei der Google Cloud Authentifizierung (DefaultCredentialsError).")
        print("Stelle sicher, dass die Umgebungsvariable GOOGLE_APPLICATION_CREDENTIALS")
        print("korrekt auf den Pfad deiner Service Account JSON-Key-Datei zeigt.")
        print("Beispiel: export GOOGLE_APPLICATION_CREDENTIALS=\"/pfad/zur/deiner/keyfile.json\"")
        print("--------------------------------------------------------------------------------------")
        raise
    except Exception as e:
        print(f"Ein unerwarteter Fehler bei der Initialisierung von Vertex AI ist aufgetreten: {e}")
        raise

def get_vertex_text_embedding(
    text: str,
    model_name: str = "text-multilingual-embedding-002",
    task_type: str = "RETRIEVAL_DOCUMENT",
    output_dimensionality: Optional[int] = None,
    project_id: Optional[str] = None,
    location: Optional[str] = None
) -> Optional[List[float]]:
    """
    Generates text embedding using Vertex AI.
    """
    # Dynamische Initialisierung, falls nicht schon geschehen
    if project_id and location and not aiplatform.initializer.global_config.project:
        try:
            init_vertex_ai(project_id, location)
        except Exception as e:
            print(f"Fehler bei der dynamischen Initialisierung von Vertex AI in get_vertex_text_embedding: {e}")
            return None

    try:
        model_to_call = TextEmbeddingModel.from_pretrained(model_name)
        instance = TextEmbeddingInput(text=text, task_type=task_type)
        embeddings = model_to_call.get_embeddings([instance])

        if embeddings and len(embeddings) > 0:
            return embeddings[0].values
        else:
            print(f"Konnte keine Embeddings für Text erhalten: {text[:100]}...")
            return None

    except Exception as e:
        print(f"Fehler beim Erhalten des Vertex AI Embeddings für '{text[:100]}...': {e}")
        return None

# Simple approximation for token counting. A more precise method would use a tokenizer,
# but for batching, character count is a reasonable and fast proxy.
# Average token length is ~3 chars for safety.
TOKEN_LIMIT = 12000  # Much more conservative limit to avoid 20k token errors

def count_tokens_approx(text: str) -> int:
    """Approximates token count based on character length."""
    if not text:
        return 0
    # Use 3.0 as average for more conservative estimation
    return max(1, int(len(text) / 3.0))

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from google.api_core import exceptions as gcp_exceptions

# Define what exceptions should trigger a retry
retry_on_vertex_error = retry_if_exception_type((
    gcp_exceptions.ResourceExhausted,  # 429
    gcp_exceptions.InternalServerError,  # 500
    gcp_exceptions.ServiceUnavailable,  # 503
    gcp_exceptions.DeadlineExceeded,    # 504
    ConnectionError,
    TimeoutError
))

import asyncio

async def get_vertex_text_embeddings_batched(
    texts: List[str],
    model_name: str = "text-multilingual-embedding-002",
    task_type: str = "RETRIEVAL_DOCUMENT",
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    concurrency_limit: int = 10
) -> List[Optional[List[float]]]:
    """
    Generates text embeddings for a list of texts using Vertex AI, with intelligent batching
    and parallel processing to stay within token and rate limits.
    """
    if project_id and location and not aiplatform.initializer.global_config.project:
        try:
            init_vertex_ai(project_id, location)
        except Exception as e:
            print(f"Fehler bei der dynamischen Initialisierung von Vertex AI: {e}")
            return [None] * len(texts)

    try:
        model = TextEmbeddingModel.from_pretrained(model_name)
    except Exception as e:
        print(f"Fehler beim Laden des Vertex AI Embedding Modells '{model_name}': {e}")
        return [None] * len(texts)

    # --- Batch Creation ---
    batches = []
    texts_with_indices = list(zip(range(len(texts)), texts))
    while texts_with_indices:
        current_batch_texts = []
        current_batch_indices = []
        current_batch_tokens = 0
        while texts_with_indices:
            index, text = texts_with_indices[0]
            text_tokens = count_tokens_approx(text)
            if text_tokens > TOKEN_LIMIT:
                print(f"WARNUNG: Text bei Index {index} ist mit ca. {text_tokens} Tokens zu lang und wird übersprungen.")
                texts_with_indices.pop(0)
                continue
            if current_batch_tokens + text_tokens > TOKEN_LIMIT and current_batch_texts:
                break
            current_batch_tokens += text_tokens
            current_batch_texts.append(text)
            current_batch_indices.append(index)
            texts_with_indices.pop(0)
        if current_batch_texts:
            batches.append((current_batch_indices, current_batch_texts, current_batch_tokens))

    # --- Parallel Processing ---
    semaphore = asyncio.Semaphore(concurrency_limit)
    all_embeddings: List[Optional[List[float]]] = [None] * len(texts)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
        retry=retry_on_vertex_error
    )
    async def _get_embeddings_for_batch(batch_texts, batch_tokens):
        print(f"Verarbeite Batch mit {len(batch_texts)} Texten (ca. {batch_tokens} Tokens)...")
        instances = [TextEmbeddingInput(text=t, task_type=task_type) for t in batch_texts]
        return await asyncio.to_thread(model.get_embeddings, instances)

    async def _process_batch(batch_indices, batch_texts, batch_tokens):
        async with semaphore:
            try:
                # Additional safety check before processing
                if batch_tokens > 15000:  # Extra conservative check
                    print(f"WARNUNG: Batch mit {batch_tokens} Tokens ist zu groß, teile auf...")
                    # Split the batch in half and process separately
                    mid = len(batch_texts) // 2
                    if mid > 0:
                        # Process first half
                        await _process_batch(batch_indices[:mid], batch_texts[:mid], 
                                           sum(count_tokens_approx(t) for t in batch_texts[:mid]))
                        # Process second half
                        await _process_batch(batch_indices[mid:], batch_texts[mid:], 
                                           sum(count_tokens_approx(t) for t in batch_texts[mid:]))
                        return
                
                embeddings = await _get_embeddings_for_batch(batch_texts, batch_tokens)
                if embeddings:
                    for i, resp in enumerate(embeddings):
                        original_idx = batch_indices[i]
                        all_embeddings[original_idx] = resp.values
            except Exception as e:
                error_msg = str(e)
                if "token count" in error_msg and "20000" in error_msg:
                    print(f"TOKEN LIMIT FEHLER: Batch (Startindex: {batch_indices[0]}) überschreitet 20k Token-Limit")
                    # Try to split the batch further
                    if len(batch_texts) > 1:
                        print(f"Teile Batch mit {len(batch_texts)} Texten auf...")
                        mid = len(batch_texts) // 2
                        await _process_batch(batch_indices[:mid], batch_texts[:mid], 
                                           sum(count_tokens_approx(t) for t in batch_texts[:mid]))
                        await _process_batch(batch_indices[mid:], batch_texts[mid:], 
                                           sum(count_tokens_approx(t) for t in batch_texts[mid:]))
                    else:
                        print(f"Einzelner Text zu lang: {len(batch_texts[0])} Zeichen")
                        all_embeddings[batch_indices[0]] = None
                else:
                    print(f"FEHLER: Batch (Startindex: {batch_indices[0]}) schlug nach mehreren Versuchen fehl: {e}")
                    # Mark embeddings for this batch as None
                    for original_idx in batch_indices:
                        all_embeddings[original_idx] = None

    tasks = [_process_batch(indices, texts_b, tokens) for indices, texts_b, tokens in batches]
    await asyncio.gather(*tasks)

    return all_embeddings

if __name__ == "__main__":
    print("Initialisiere Vertex AI (Beispiel)...")
    try:
        init_vertex_ai(
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT"), 
            location=os.getenv("GOOGLE_CLOUD_LOCATION")
        )

        test_texts = [
            "Das ist ein Testdokument.",
            "Ein weiteres Dokument für die semantische Suche.",
            "Wie funktioniert das Embedding von Texten?",
            "Hallo Welt!",
            "Kurzer Text.",
            "Noch ein Text, um das Batching zu testen."
        ]

        print(f"\nTeste einzelnen Embedding-Aufruf mit 'text-embedding-004':")
        single_embedding = get_vertex_text_embedding(
            test_texts[0],
            model_name="text-embedding-004",
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        )
        if single_embedding:
            print(f"Embedding für '{test_texts[0]}': Dimension {len(single_embedding)}, erste 3 Werte: {single_embedding[:3]}")
        else:
            print(f"Fehler beim Erstellen des einzelnen Embeddings für '{test_texts[0]}'.")

        print(f"\nTeste Batch Embedding-Aufruf mit 'text-embedding-004':")
        batch_embeddings = get_vertex_text_embeddings_batched(
            test_texts,
            model_name="text-embedding-004",
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        )

        for text, emb in zip(test_texts, batch_embeddings):
            if emb:
                print(f"Text: '{text}', Embedding-Dim: {len(emb)}, Erste 3: {emb[:3]}")
            else:
                print(f"Text: '{text}', Embedding: Fehler")

    except Exception as e:
        print(f"Fehler im Beispielaufruf: {e}")
        print("Stelle sicher, dass die Google Cloud Authentifizierung und Projektkonfiguration korrekt sind.")
