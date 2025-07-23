"""
insert_docs_streamlit.py
-----------------------
Angepasste Version von insert_docs.py für Streamlit Community Cloud.
Nutzt Modal.com für Crawling statt lokales Crawl4AI.
"""

import asyncio
import re
import os
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import streamlit as st

# Import des Modal Crawler Clients
from crawler_client import CrawlerClient

# Import der bestehenden Utility-Funktionen
from utils import get_chroma_client, get_or_create_collection, add_documents_to_collection
from vertex_ai_utils import get_vertex_text_embeddings_batched, init_vertex_ai

def smart_chunk_markdown(markdown: str, max_len: int = 1500, overlap_len: int = 150) -> List[str]:
    """
    Hierarchically splits markdown by #, ##, ### headers.
    If any resulting chunk is still > max_len, it splits that chunk by length with overlap.
    Ensures all chunks are < max_len.
    """
    if overlap_len >= max_len:
        raise ValueError("Overlap length must be smaller than max length")

    # Helper to split a text block strictly by length with overlap
    def split_by_length(text_block, length, overlap):
        if overlap >= length:
            raise ValueError("Overlap must be smaller than chunk length")
             
        chunks = []
        start = 0
        text_len = len(text_block)
        
        while start < text_len:
            end = min(start + length, text_len)
            chunk = text_block[start:end].strip()
            if chunk:
                chunks.append(chunk)
                
            # Calculate the start of the next chunk
            next_start = start + length - overlap
            
            # If the next start position is the same or behind the current one,
            # or if we reached the end, break the loop.
            if next_start <= start or end == text_len:
                break
                
            start = next_start
            
        # Filter out potential empty strings resulting from strip()
        return [c for c in chunks if c]

    # Helper to split by header pattern, keeping headers with their content
    def split_by_header(md_content, header_level):
        # Regex to find headers of the specified level (e.g., ## Header)
        pattern = re.compile(rf'^(?={header_level} )', re.MULTILINE)
        
        parts = []
        matches = list(pattern.finditer(md_content))

        # Add content before the first header (if any)
        if not matches or matches[0].start() > 0:
            first_part = md_content[0:matches[0].start() if matches else len(md_content)].strip()
            if first_part:
                parts.append(first_part)
        
        # Process each header section
        for i, match in enumerate(matches):
            start = match.start()
            # Determine the end of the current section
            end = matches[i+1].start() if i + 1 < len(matches) else len(md_content)
            section = md_content[start:end].strip()
            if section:
                parts.append(section)
                
        # If no headers found, return the whole content as one part
        if not parts and md_content:
            return [md_content]
            
        return parts

    final_chunks = []
    
    # Start splitting by H1
    h1_sections = split_by_header(markdown, '#')
    for h1_sec in h1_sections:
        # Split by H2
        h2_sections = split_by_header(h1_sec, '##')
        if len(h2_sections) == 1 and h2_sections[0] == h1_sec:
             h2_sections_to_process = [h1_sec]
        else:
             h2_sections_to_process = h2_sections
             
        for h2_sec in h2_sections_to_process:
            # Split by H3
            h3_sections = split_by_header(h2_sec, '###')
            if len(h3_sections) == 1 and h3_sections[0] == h2_sec:
                initial_chunks = [h2_sec]
            else:
                 initial_chunks = h3_sections
                 
            # Process the resulting chunks
            for chunk in initial_chunks:
                if len(chunk) > max_len:
                    # If header splitting wasn't enough, split by length WITH OVERLAP
                    final_chunks.extend(split_by_length(chunk, max_len, overlap_len))
                elif chunk:
                    final_chunks.append(chunk)

    # Handle case where the entire markdown had no headers at all
    if not final_chunks and markdown:
         if len(markdown) > max_len:
             final_chunks.extend(split_by_length(markdown, max_len, overlap_len))
         else:
             final_chunks.append(markdown)
             
    return final_chunks

def is_sitemap(url: str) -> bool:
    """Prüft, ob eine URL eine Sitemap ist."""
    return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path

def is_txt(url: str) -> bool:
    """Prüft, ob eine URL eine Textdatei ist."""
    return url.endswith('.txt') or url.endswith('.md')

def get_expected_embedding_dimensions(model_name: str) -> int:
    """Gibt die erwarteten Embedding-Dimensionen für ein Vertex AI Modell zurück."""
    model_dimensions = {
        "text-multilingual-embedding-002": 768,
        "gemini-embedding-001": 3072,
        "text-embedding-005": 768,
        "text-embedding-004": 768,  # Legacy model
    }
    return model_dimensions.get(model_name, 768)  # Default zu 768

def get_embedding_model_info(model_name: str) -> Dict[str, Any]:
    """Gibt Informationen über ein Vertex AI Embedding-Modell zurück."""
    model_info = {
        "text-multilingual-embedding-002": {
            "dimensions": 768,
            "description": "Multilingual (768D)",
            "languages": "Mehrsprachig",
            "max_tokens": 2048
        },
        "gemini-embedding-001": {
            "dimensions": 3072,
            "description": "Gemini (3072D)",
            "languages": "Mehrsprachig + Code",
            "max_tokens": 2048
        },
        "text-embedding-005": {
            "dimensions": 768,
            "description": "English + Code (768D)",
            "languages": "Englisch + Code",
            "max_tokens": 2048
        }
    }
    return model_info.get(model_name, {
        "dimensions": 768,
        "description": f"Unknown model ({model_name})",
        "languages": "Unknown",
        "max_tokens": 2048
    })

def extract_section_info(chunk: str) -> Dict[str, Any]:
    """Extracts headers and stats from a chunk."""
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }

def sanitize_collection_name(name: str) -> str:
    """Sanitizes a string to be a valid ChromaDB collection name."""
    if not name:
        return "default-collection"
    
    # Make it lowercase
    name = name.lower()
    # Replace spaces and invalid characters with hyphens
    name = re.sub(r'[\s_.]+', '-', name)
    name = re.sub(r'[^a-z0-9-]', '', name)
    
    # Ensure it's not too long (ChromaDB has a limit of 63)
    name = name[:60]
    
    # Ensure it doesn't start or end with a hyphen
    name = name.strip('-')
    
    # ChromaDB requires length between 3 and 63
    if len(name) < 3:
        name = f"{name}-kb"
    
    # Final check
    if not name:
        return "default-kb"
        
    return name

def generate_collection_name_from_url(url: str) -> str:
    """Generates a ChromaDB-compatible collection name from a URL."""
    if not url or not url.strip():
        return "default-collection"

    try:
        parsed_url = urlparse(url)
        if not parsed_url.netloc:
            # Fallback for local file paths
            base_name = os.path.basename(url)
            name, _ = os.path.splitext(base_name)
        else:
            # Start with the domain name
            name = parsed_url.netloc
            # Add path, replacing slashes with hyphens
            path = parsed_url.path
            if path and path != "/":
                name += path.replace('/', '-').rstrip('-')
    except Exception:
        return "default-collection"

    return sanitize_collection_name(name)

class IngestionProgress:
    """Helper class to track and display ingestion progress."""
    
    def __init__(self):
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.current_step = 0
        self.total_steps = 5
    
    def update(self, step: int, message: str):
        """Update progress bar and status message."""
        self.current_step = step
        progress = int((step / self.total_steps) * 100)
        self.progress_bar.progress(progress)
        self.status_text.text(f"Schritt {step}/{self.total_steps}: {message}")
    
    def complete(self, message: str = "Ingestion abgeschlossen!"):
        """Mark ingestion as complete."""
        self.progress_bar.progress(100)
        self.status_text.text(message)

async def run_ingestion_with_modal(
    url: str,
    collection_name: Optional[str],
    crawler_client: CrawlerClient,
    chroma_client,
    chunk_size: int = 1500,
    chunk_overlap: int = 150,
    max_depth: int = 3,
    max_concurrent: int = 5,
    limit: Optional[int] = 100,
    progress: Optional[IngestionProgress] = None,
    auto_reduce: bool = True,
    max_chunks: Optional[int] = None
) -> Dict[str, Any]:
    """
    Führt die komplette Ingestion mit Modal.com Crawler durch.
    
    Args:
        url: URL zum Crawlen
        collection_name: Name der ChromaDB Collection
        crawler_client: Modal.com Crawler Client
        chroma_client: ChromaDB Client
        chunk_size: Größe der Text-Chunks
        chunk_overlap: Überlappung zwischen Chunks
        max_depth: Maximale Crawling-Tiefe
        max_concurrent: Maximale parallele Crawling-Prozesse
        limit: Maximale Anzahl URLs
        progress: Progress-Tracker (optional)
    
    Returns:
        Dictionary mit Ingestion-Ergebnissen
    """
    try:
        # Schritt 1: Collection Name generieren
        if progress:
            progress.update(1, "Bereite Collection vor...")
        
        if collection_name:
            collection_name = sanitize_collection_name(collection_name)
        else:
            collection_name = generate_collection_name_from_url(url)
        
        print(f"Using collection name: '{collection_name}'")
        
        # Schritt 2: Crawling durchführen
        if progress:
            progress.update(2, "Crawle Webseiten...")
        
        # URL-Typ erkennen und entsprechende Crawling-Methode wählen
        if is_txt(url):
            print(f"Detected text file: {url}")
            crawl_result = await crawler_client.crawl_single(url)
            crawl_results = [crawl_result] if crawl_result.get("success") else []
        elif is_sitemap(url):
            print(f"Detected sitemap: {url}")
            crawl_result = await crawler_client.crawl_sitemap(url, max_concurrent=max_concurrent)
            crawl_results = crawl_result.get("results", []) if crawl_result else []
        else:
            print(f"Detected regular URL: {url}")
            crawl_result = await crawler_client.crawl_recursive(
                start_url=url,
                max_depth=max_depth,
                max_concurrent=max_concurrent,
                limit=limit
            )
            crawl_results = crawl_result.get("results", []) if crawl_result else []
        
        # Filter erfolgreiche Crawling-Ergebnisse
        successful_results = [r for r in crawl_results if r.get("success", False) and r.get("markdown")]
        
        if not successful_results:
            raise ValueError("Keine erfolgreichen Crawling-Ergebnisse gefunden")
        
        print(f"Successfully crawled {len(successful_results)} pages")
        
        # Schritt 3: Text in Chunks aufteilen
        if progress:
            progress.update(3, f"Teile {len(successful_results)} Dokumente in Chunks auf...")
        
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for i, result in enumerate(successful_results):
            markdown_content = result.get('markdown', '')
            url_source = result.get('url', f'unknown_{i}')
            
            if markdown_content:
                chunks = smart_chunk_markdown(markdown_content, max_len=chunk_size, overlap_len=chunk_overlap)
                
                for j, chunk_text in enumerate(chunks):
                    doc_id = f"doc_{sanitize_collection_name(url_source)}_{j}"
                    metadata = extract_section_info(chunk_text)
                    metadata['url'] = url_source
                    metadata['chunk_index'] = j
                    metadata['total_chunks'] = len(chunks)
                    
                    all_chunks.append(chunk_text)
                    all_metadatas.append(metadata)
                    all_ids.append(doc_id)
        
        if not all_chunks:
            raise ValueError("Keine Text-Chunks konnten erstellt werden")
        
        print(f"Created {len(all_chunks)} chunks from {len(successful_results)} documents")
        
        # Schritt 4: Embeddings generieren (falls Vertex AI verfügbar)
        if progress:
            progress.update(4, f"Generiere Embeddings für {len(all_chunks)} Chunks...")
        
        all_embeddings = None
        
        # Prüfe, ob Vertex AI verfügbar ist
        vertex_project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or st.secrets.get("GOOGLE_CLOUD_PROJECT")
        vertex_location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1") or st.secrets.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        
        # Überspringe Vertex AI wenn Platzhalter-Werte verwendet werden
        if vertex_project_id and vertex_project_id != "your-gcp-project-id":
            try:
                # Setup Google Cloud Credentials falls noch nicht geschehen
                if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                    print("Setting up Google Cloud credentials from Streamlit secrets...")
                    creds_json_b64 = st.secrets.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
                    if creds_json_b64:
                        import base64
                        import tempfile
                        # Dekodiere und speichere temporär
                        creds_json = base64.b64decode(creds_json_b64).decode('utf-8')
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w') as temp:
                            temp.write(creds_json)
                            temp_filename = temp.name
                        
                        # Setze die Umgebungsvariable
                        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_filename
                        print("✅ Google Cloud Credentials successfully loaded from secrets")
                        st.info("✅ Google Cloud Credentials geladen")
                    else:
                        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS_JSON not found in secrets")
                
                # Embedding-Modell konfigurieren (aus Umgebungsvariablen oder Standard)
                embedding_model = os.environ.get("VERTEX_EMBEDDING_MODEL", "text-multilingual-embedding-002")
                if hasattr(st, 'secrets') and st.secrets.get("VERTEX_EMBEDDING_MODEL"):
                    embedding_model = st.secrets.get("VERTEX_EMBEDDING_MODEL")
                
                model_info = get_embedding_model_info(embedding_model)
                expected_dimensions = model_info["dimensions"]
                
                print(f"Initializing Vertex AI for embeddings with model: {embedding_model}")
                print(f"Expected embedding dimensions: {expected_dimensions}")
                st.info(f"🔧 Verwende Vertex AI Modell: {embedding_model} ({model_info['description']})")
                
                init_vertex_ai(project_id=vertex_project_id, location=vertex_location)
                
                print(f"Generating embeddings for {len(all_chunks)} chunks...")
                all_embeddings_vertex = await get_vertex_text_embeddings_batched(
                    texts=all_chunks,
                    model_name=embedding_model,
                    task_type="RETRIEVAL_DOCUMENT",
                    project_id=vertex_project_id,
                    location=vertex_location
                )
                
                # Filter failed embeddings
                failed_count = all_embeddings_vertex.count(None)
                if failed_count > 0:
                    print(f"WARNING: {failed_count} embeddings failed")
                    
                    # Filter out failed embeddings
                    filtered_chunks = []
                    filtered_metadatas = []
                    filtered_ids = []
                    filtered_embeddings = []
                    
                    for i, embedding in enumerate(all_embeddings_vertex):
                        if embedding is not None:
                            filtered_chunks.append(all_chunks[i])
                            filtered_metadatas.append(all_metadatas[i])
                            filtered_ids.append(all_ids[i])
                            filtered_embeddings.append(embedding)
                    
                    all_chunks = filtered_chunks
                    all_metadatas = filtered_metadatas
                    all_ids = filtered_ids
                    all_embeddings = filtered_embeddings
                else:
                    all_embeddings = all_embeddings_vertex
                
                embedding_dim = len(all_embeddings[0]) if all_embeddings else 0
                print(f"Successfully generated {len(all_embeddings)} embeddings with {embedding_dim} dimensions")
                
                # Validiere, dass die Dimensionen korrekt sind
                if embedding_dim != expected_dimensions:
                    print(f"WARNING: Expected {expected_dimensions} dimensions but got {embedding_dim}")
                    st.warning(f"⚠️ Unerwartete Embedding-Dimensionen: Erwartet {expected_dimensions}, erhalten {embedding_dim}")
                
                st.success(f"✅ Vertex AI Embeddings generiert: {len(all_embeddings)} Embeddings mit {embedding_dim} Dimensionen")
                
            except Exception as e:
                print(f"Vertex AI embedding generation failed: {e}")
                st.error(f"❌ Vertex AI Embeddings fehlgeschlagen: {str(e)}")
                st.info("🔄 Verwende ChromaDB Standard-Embeddings (384 Dimensionen)")
                all_embeddings = None  # ChromaDB wird Standard-Embeddings verwenden
                
                # Wichtig: Bei Vertex AI Fehlern müssen wir sicherstellen, dass bestehende Collections
                # mit Vertex AI Embeddings gelöscht werden, da sie inkompatibel sind
                try:
                    existing_collections = [c.name for c in chroma_client.list_collections()]
                    if collection_name in existing_collections:
                        print(f"Deleting existing collection '{collection_name}' due to embedding type change")
                        chroma_client.delete_collection(name=collection_name)
                        st.warning(f"⚠️ Bestehende Collection '{collection_name}' gelöscht (Embedding-Typ geändert)")
                except Exception as delete_error:
                    print(f"Could not delete existing collection: {delete_error}")
                    
        else:
            print("No valid Vertex AI configuration found, using ChromaDB default embeddings")
            st.info("ℹ️ Verwende ChromaDB Standard-Embeddings (384 Dimensionen) - keine Google Cloud Konfiguration")
            
            # Auch hier: Bestehende Vertex AI Collections löschen
            try:
                existing_collections = [c.name for c in chroma_client.list_collections()]
                if collection_name in existing_collections:
                    # Prüfe ob die Collection Vertex AI Embeddings hat
                    test_collection = chroma_client.get_collection(name=collection_name)
                    compatibility = check_embedding_compatibility(test_collection, None)
                    if compatibility.get("expected_dimension") and compatibility["expected_dimension"] > 384:
                        print(f"Deleting existing Vertex AI collection '{collection_name}'")
                        chroma_client.delete_collection(name=collection_name)
                        st.warning(f"⚠️ Bestehende Vertex AI Collection '{collection_name}' gelöscht")
            except Exception as delete_error:
                print(f"Could not check/delete existing collection: {delete_error}")
        
        # Schritt 5: Memory-Check und ChromaDB-Speicherung
        if progress:
            progress.update(5, f"Speichere {len(all_chunks)} Chunks in ChromaDB...")
        
        # Intelligentes Memory-Management für Streamlit Cloud
        chunk_count = len(all_chunks)
        
        # Realistischere Memory-Schätzung basierend auf tatsächlicher Chunk-Größe
        avg_chunk_size = sum(len(chunk) for chunk in all_chunks) / len(all_chunks) if all_chunks else 0
        estimated_memory_mb = (chunk_count * avg_chunk_size * 2) / (1024 * 1024)  # Text + Embeddings + Overhead
        
        print(f"Memory estimate: {estimated_memory_mb:.1f}MB for {chunk_count} chunks (avg size: {avg_chunk_size:.0f} chars)")
        
        # Chunk-Reduktion basierend auf Benutzer-Einstellungen oder Memory-Limits
        original_chunk_count = chunk_count
        reduction_applied = False
        
        # Benutzer-definiertes Chunk-Limit
        if max_chunks and chunk_count > max_chunks:
            st.info(f"🔢 Reduziere auf benutzer-definiertes Limit: {max_chunks} Chunks")
            target_chunks = max_chunks
            reduction_applied = True
        # Automatische Memory-basierte Reduktion
        elif auto_reduce and (chunk_count > 5000 or estimated_memory_mb > 800):
            if estimated_memory_mb > 800:
                target_chunks = int(800 * chunk_count / estimated_memory_mb)
                st.warning(f"🔄 Auto-Reduktion: {target_chunks} Chunks um Memory-Limit einzuhalten")
                reduction_applied = True
            elif chunk_count > 5000:
                target_chunks = 5000
                st.warning(f"🔄 Auto-Reduktion: {target_chunks} Chunks für bessere Performance")
                reduction_applied = True
        
        if reduction_applied:
            # Intelligentes Sampling: Behalte die besten Chunks
            chunk_scores = []
            for i, chunk in enumerate(all_chunks):
                # Score basierend auf Länge, Struktur und Informationsgehalt
                score = len(chunk) * (1 + chunk.count('.') * 0.1 + chunk.count('#') * 0.2)
                # Bevorzuge Chunks mit Headers und strukturiertem Content
                if any(header in chunk for header in ['##', '###', '**']):
                    score *= 1.2
                chunk_scores.append((score, i))
            
            # Sortiere nach Score und nimm die besten
            chunk_scores.sort(reverse=True)
            selected_indices = [idx for _, idx in chunk_scores[:target_chunks]]
            selected_indices.sort()  # Behalte ursprüngliche Reihenfolge
            
            # Filtere Arrays
            all_chunks = [all_chunks[i] for i in selected_indices]
            all_metadatas = [all_metadatas[i] for i in selected_indices]
            all_ids = [all_ids[i] for i in selected_indices]
            if all_embeddings:
                all_embeddings = [all_embeddings[i] for i in selected_indices]
            
            chunk_count = len(all_chunks)
            estimated_memory_mb = (chunk_count * avg_chunk_size * 2) / (1024 * 1024)
            
            st.success(f"✅ Dataset reduziert: {original_chunk_count} → {chunk_count} Chunks (~{estimated_memory_mb:.0f}MB)")
        
        # Finale Memory-Prüfung mit höherem Limit
        if estimated_memory_mb > 1200:  # Erhöhtes Limit, da Schätzung konservativer ist
            st.error(f"❌ Memory-Limit überschritten: {estimated_memory_mb:.0f}MB für {chunk_count} Chunks")
            st.info("💡 Versuche: Kleinere Chunk-Größe, weniger URLs, oder verwende 'Einzelne URL' statt 'Sitemap'")
            raise ValueError(f"Memory limit exceeded: {estimated_memory_mb:.0f}MB estimated for {chunk_count} chunks")
        elif estimated_memory_mb > 800:
            st.warning(f"⚠️ Hoher Speicherverbrauch: ~{estimated_memory_mb:.0f}MB für {chunk_count} Chunks")
        
        print(f"Final memory estimate: {estimated_memory_mb:.1f}MB for {chunk_count} chunks")
        
        # Collection erstellen oder laden mit korrekter Embedding-Funktion
        from chromadb.utils import embedding_functions
        
        # Bestimme die richtige Embedding-Funktion basierend auf verfügbaren Embeddings
        embedding_function = None
        if all_embeddings is None:
            # Verwende ChromaDB's Standard-Embedding-Funktion (384 Dimensionen)
            embedding_function = embedding_functions.DefaultEmbeddingFunction()
            print("Using ChromaDB default embedding function (384 dimensions)")
        else:
            # Wenn Vertex AI Embeddings verfügbar sind (768 Dimensionen), verwende keine Embedding-Funktion
            # ChromaDB wird die bereitgestellten Embeddings direkt verwenden
            embedding_function = None
            print(f"Using provided Vertex AI embeddings ({len(all_embeddings[0])} dimensions)")
        
        # Versuche Collection zu erstellen/laden, bei Dimensionen-Konflikt neu erstellen
        from utils import check_embedding_compatibility
        
        collection = None
        try:
            collection = get_or_create_collection(
                client=chroma_client,
                collection_name=collection_name,
                embedding_function=embedding_function
            )
            
            # Prüfe Embedding-Kompatibilität
            test_embedding = all_embeddings[0] if all_embeddings else None
            compatibility = check_embedding_compatibility(collection, test_embedding)
            
            if not compatibility["compatible"]:
                print(f"Embedding dimension incompatibility detected: {compatibility['reason']}")
                st.warning(f"⚠️ Embedding-Dimensionen-Konflikt: {compatibility['reason']}")
                raise ValueError(f"Dimension conflict: {compatibility['reason']}")
            else:
                print(f"Collection compatibility check passed: {compatibility['reason']}")
                
        except Exception as e:
            if "dimension" in str(e).lower() or "Dimension conflict" in str(e):
                print(f"Embedding dimension conflict detected: {e}")
                st.warning(f"⚠️ Embedding-Dimensionen-Konflikt erkannt. Erstelle Collection neu...")
                
                # Collection löschen und neu erstellen
                try:
                    chroma_client.delete_collection(name=collection_name)
                    print(f"Deleted existing collection '{collection_name}' due to dimension conflict")
                except Exception as delete_error:
                    print(f"Could not delete collection (may not exist): {delete_error}")
                
                # Neue Collection mit korrekter Embedding-Funktion erstellen
                collection = get_or_create_collection(
                    client=chroma_client,
                    collection_name=collection_name,
                    embedding_function=embedding_function
                )
                print(f"Created new collection '{collection_name}' with correct embedding dimensions")
                st.success(f"✅ Collection '{collection_name}' wurde mit korrekten Dimensionen neu erstellt")
            else:
                raise e
        
        # Dokumente in Batches hinzufügen für bessere Memory-Performance
        if all_chunks:  # Nur hinzufügen wenn Chunks vorhanden sind
            # Dynamische Batch-Größe basierend auf Chunk-Anzahl
            if chunk_count > 1000:
                batch_size = 50  # Kleinere Batches für große Datasets
                st.info(f"ℹ️ Verwende kleinere Batch-Größe ({batch_size}) für {chunk_count} Chunks")
            elif chunk_count > 500:
                batch_size = 75
            else:
                batch_size = 100
            
            print(f"Adding {chunk_count} documents in batches of {batch_size}")
            
            add_documents_to_collection(
                collection=collection,
                ids=all_ids,
                documents=all_chunks,
                embeddings=all_embeddings,  # None = ChromaDB Standard-Embeddings
                metadatas=all_metadatas,
                batch_size=batch_size
            )
            
            # Memory-Status nach dem Hinzufügen
            final_count = collection.count()
            print(f"Collection now contains {final_count} documents")
            if final_count > 5000:
                st.warning(f"⚠️ Collection enthält {final_count} Dokumente. Performance könnte beeinträchtigt sein.")
                
        else:
            raise ValueError("Keine Chunks zum Speichern verfügbar")
        
        if progress:
            progress.complete(f"Erfolgreich {len(all_chunks)} Chunks gespeichert!")
        
        return {
            "success": True,
            "collection_name": collection_name,
            "documents_crawled": len(successful_results),
            "chunks_created": len(all_chunks),
            "embeddings_generated": len(all_embeddings) if all_embeddings else 0,
            "failed_crawls": len(crawl_results) - len(successful_results)
        }
        
    except Exception as e:
        error_msg = f"Ingestion failed: {str(e)}"
        print(error_msg)
        if progress:
            progress.status_text.text(f"❌ {error_msg}")
        raise e

# RAG Query-Funktion mit korrekten Embeddings
async def query_collection_with_embeddings(
    collection,
    query_text: str,
    n_results: int = 5
) -> Dict[str, Any]:
    """
    Query eine Collection mit den korrekten Embedding-Dimensionen.
    Verwendet Vertex AI Embeddings falls verfügbar, sonst ChromaDB Default.
    """
    try:
        # Prüfe, ob Vertex AI verfügbar ist
        vertex_project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or st.secrets.get("GOOGLE_CLOUD_PROJECT")
        vertex_location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1") or st.secrets.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        
        query_embeddings = None
        
        if vertex_project_id and vertex_project_id != "your-gcp-project-id":
            try:
                # Setup Google Cloud Credentials falls noch nicht geschehen
                if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                    creds_json_b64 = st.secrets.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
                    if creds_json_b64:
                        import base64
                        import tempfile
                        creds_json = base64.b64decode(creds_json_b64).decode('utf-8')
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w') as temp:
                            temp.write(creds_json)
                            temp_filename = temp.name
                        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_filename
                
                # Embedding-Modell konfigurieren
                embedding_model = os.environ.get("VERTEX_EMBEDDING_MODEL", "text-multilingual-embedding-002")
                if hasattr(st, 'secrets') and st.secrets.get("VERTEX_EMBEDDING_MODEL"):
                    embedding_model = st.secrets.get("VERTEX_EMBEDDING_MODEL")
                
                print(f"Generating query embedding with Vertex AI model: {embedding_model}")
                
                # Vertex AI für Query initialisieren
                init_vertex_ai(project_id=vertex_project_id, location=vertex_location)
                
                # Query-Embedding generieren
                from vertex_ai_utils import get_vertex_text_embedding
                query_embedding = await asyncio.to_thread(
                    get_vertex_text_embedding,
                    text=query_text,
                    model_name=embedding_model,
                    task_type="RETRIEVAL_QUERY",  # Wichtig: QUERY statt DOCUMENT
                    project_id=vertex_project_id,
                    location=vertex_location
                )
                
                if query_embedding:
                    query_embeddings = [query_embedding]
                    print(f"Generated Vertex AI query embedding with {len(query_embedding)} dimensions")
                else:
                    print("Vertex AI query embedding failed, falling back to ChromaDB default")
                    
            except Exception as e:
                print(f"Vertex AI query embedding failed: {e}")
                st.warning(f"⚠️ Vertex AI Query-Embedding fehlgeschlagen, verwende ChromaDB Default")
        
        # Query ausführen
        if query_embeddings:
            # Mit Vertex AI Embeddings
            results = collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            print("Query executed with Vertex AI embeddings")
        else:
            # Mit ChromaDB Default Embeddings
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            print("Query executed with ChromaDB default embeddings")
        
        return results
        
    except Exception as e:
        print(f"Query failed: {e}")
        raise e

def query_collection_sync(collection, query_text: str, n_results: int = 5) -> Dict[str, Any]:
    """Synchroner Wrapper für query_collection_with_embeddings."""
    return asyncio.run(query_collection_with_embeddings(collection, query_text, n_results))

def clean_wikipedia_text(text: str) -> str:
    """Bereinigt Wikipedia-Markup und andere Störungen aus dem Text."""
    if not text:
        return text
    
    # Wikipedia-spezifische Bereinigungen
    text = re.sub(r'Datei:[^)]*\)', '', text)  # Entferne "Datei:xyz.svg)"
    text = re.sub(r'\[Bearbeiten \| Quelltext bearbeiten\]', '', text)  # Wikipedia-Edit-Links
    text = re.sub(r'\[\d+\]', '', text)  # Referenz-Nummern [1], [2], etc.
    text = re.sub(r'\[.*?\]', '', text)  # Andere eckige Klammern
    text = re.sub(r'→ Hauptartikel:.*?\n', '', text)  # Hauptartikel-Verweise
    text = re.sub(r'Siehe auch:.*?\n', '', text)  # Siehe-auch-Verweise
    text = re.sub(r'Weblinks.*$', '', text, flags=re.DOTALL)  # Weblinks-Sektion
    text = re.sub(r'Literatur.*$', '', text, flags=re.DOTALL)  # Literatur-Sektion
    text = re.sub(r'Einzelnachweise.*$', '', text, flags=re.DOTALL)  # Einzelnachweise
    
    # Allgemeine Bereinigungen
    text = re.sub(r'\n\s*\n', '\n', text)  # Mehrfache Leerzeilen
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)  # Leerzeichen am Zeilenanfang/-ende
    text = text.strip()
    
    return text

def generate_rag_response(query: str, search_results: Dict[str, Any], collection_name: str) -> str:
    """
    Generiert eine intelligente RAG-Antwort basierend auf den Suchergebnissen.
    """
    try:
        documents = search_results.get('documents', [[]])[0]
        metadatas = search_results.get('metadatas', [[]])[0]
        distances = search_results.get('distances', [[]])[0]
        
        if not documents:
            return "❌ Keine relevanten Informationen gefunden."
        
        # Bereinige und filtere die besten Chunks
        cleaned_chunks = []
        sources = set()
        
        for i, (doc, meta, distance) in enumerate(zip(documents[:3], metadatas[:3], distances[:3])):
            # Text bereinigen
            cleaned_doc = clean_wikipedia_text(doc)
            
            # Nur relevante Chunks verwenden (nicht zu kurz, nicht nur Referenzen)
            if len(cleaned_doc) > 50 and not cleaned_doc.startswith('↑'):
                cleaned_chunks.append({
                    'text': cleaned_doc,
                    'url': meta.get('url', ''),
                    'distance': distance,
                    'headers': meta.get('headers', '')
                })
                
                if meta.get('url'):
                    sources.add(meta['url'])
        
        if not cleaned_chunks:
            return "❌ Keine verwertbaren Informationen nach der Bereinigung gefunden."
        
        # Erstelle eine strukturierte Antwort
        response_parts = []
        response_parts.append(f"**Antwort basierend auf '{collection_name}':**\n")
        
        # Hauptantwort aus dem besten Chunk
        best_chunk = cleaned_chunks[0]
        main_text = best_chunk['text']
        
        # Kürze den Text intelligent (an Satzenden)
        if len(main_text) > 800:
            sentences = main_text.split('. ')
            truncated = []
            current_length = 0
            
            for sentence in sentences:
                if current_length + len(sentence) > 800:
                    break
                truncated.append(sentence)
                current_length += len(sentence)
            
            main_text = '. '.join(truncated)
            if not main_text.endswith('.'):
                main_text += '.'
        
        response_parts.append(main_text)
        
        # Zusätzliche relevante Informationen
        if len(cleaned_chunks) > 1:
            response_parts.append("\n**Weitere relevante Informationen:**")
            for chunk in cleaned_chunks[1:2]:  # Nur einen zusätzlichen Chunk
                additional_text = chunk['text'][:300]
                if len(chunk['text']) > 300:
                    # Finde das letzte vollständige Wort
                    last_space = additional_text.rfind(' ')
                    if last_space > 200:
                        additional_text = additional_text[:last_space] + '...'
                response_parts.append(f"• {additional_text}")
        
        # Quellen hinzufügen
        if sources:
            response_parts.append("\n**Quellen:**")
            for i, source in enumerate(sorted(sources), 1):
                response_parts.append(f"{i}. {source}")
        
        # Debug-Info (optional, kann später entfernt werden)
        response_parts.append(f"\n*Gefunden: {len(documents)} Chunks, verwendet: {len(cleaned_chunks)} nach Bereinigung*")
        
        return '\n'.join(response_parts)
        
    except Exception as e:
        print(f"Error generating RAG response: {e}")
        return f"❌ Fehler bei der Antwortgenerierung: {str(e)}"

# Synchroner Wrapper für Streamlit
def run_ingestion_sync(
    url: str,
    collection_name: Optional[str],
    crawler_client: CrawlerClient,
    chroma_client,
    **kwargs
) -> Dict[str, Any]:
    """Synchroner Wrapper für run_ingestion_with_modal."""
    return asyncio.run(run_ingestion_with_modal(
        url=url,
        collection_name=collection_name,
        crawler_client=crawler_client,
        chroma_client=chroma_client,
        **kwargs
    ))

if __name__ == "__main__":
    print("insert_docs_streamlit.py - Modal.com Integration")
    print("Verwende diese Datei als Modul in der Streamlit-App")