"""
insert_docs.py
--------------
Command-line utility to crawl any URL using Crawl4AI, detect content type (sitemap, .txt, or regular page),
use the appropriate crawl method, chunk the resulting Markdown into <1000 character blocks by header hierarchy,
and insert all chunks into ChromaDB with metadata.

Usage:
    python insert_docs.py <URL> [--collection ...] [--db-dir ...] [--embedding-model ...]
"""
import argparse
import sys
import re
import asyncio
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from utils import get_chroma_client, get_or_create_collection, add_documents_to_collection

from chromadb.utils import embedding_functions
from vertex_ai_utils import get_vertex_text_embeddings_batched, init_vertex_ai # Hinzugefügt
import os # Hinzugefügt für Umgebungsvariablen

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
             # This case should technically be caught by the check at the start
             # but double-checking here for safety.
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
        # It captures the header line itself and looks ahead for the next header or end of string
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
        if len(h2_sections) == 1 and h2_sections[0] == h1_sec: # No effective H2 split
             h2_sections_to_process = [h1_sec] # Process the H1 section directly for H3
        else:
             h2_sections_to_process = h2_sections
             
        for h2_sec in h2_sections_to_process:
            # Split by H3
            h3_sections = split_by_header(h2_sec, '###')
            if len(h3_sections) == 1 and h3_sections[0] == h2_sec: # No effective H3 split
                initial_chunks = [h2_sec] # Add the H2/H1 section as is
            else:
                 initial_chunks = h3_sections
                 
            # Process the resulting chunks (either H3 sections or the fallback H2/H1 section)
            for chunk in initial_chunks:
                if len(chunk) > max_len:
                    # If header splitting wasn't enough or failed, split by length WITH OVERLAP
                    final_chunks.extend(split_by_length(chunk, max_len, overlap_len))
                elif chunk: # Add non-empty chunks
                    final_chunks.append(chunk)

    # Handle case where the entire markdown had no headers at all
    if not final_chunks and markdown:
         if len(markdown) > max_len:
             final_chunks.extend(split_by_length(markdown, max_len, overlap_len))
         else:
             final_chunks.append(markdown)
             
    return final_chunks

def is_sitemap(url: str) -> bool:
    return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path

def is_txt(url: str) -> bool:
    return url.endswith('.txt')

async def crawl_recursive_internal_links(start_urls, max_depth=20, max_concurrent=10, limit: Optional[int] = None):
    """Recursive crawl using logic from 5-crawl_recursive_internal_links.py.
    Returns list of dicts with url and cleaned markdown.
    Added limit parameter and content cleaning directly on markdown.
    """
    print(f"Starting crawl from: {start_urls}, Max Depth: {max_depth}, Max Concurrent: {max_concurrent}, Limit: {limit}")
    visited = set()
    current_urls = set(start_urls)
    results_all = []
    browser_config = BrowserConfig(headless=True)
    # Revert CrawlerRunConfig to default (or previous state)
    crawl_config = CrawlerRunConfig() 

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for depth in range(max_depth):
            if not current_urls:
                break
            
            # Check limit before starting new level
            if limit is not None and len(visited) >= limit:
                print(f"Crawl limit ({limit}) reached. Stopping.")
                break

            next_level_urls = set()
            tasks = []
            urls_to_crawl_this_level = list(current_urls - visited)
            
            # Apply limit within the current level as well
            if limit is not None:
                remaining_slots = limit - len(visited)
                urls_to_crawl_this_level = urls_to_crawl_this_level[:remaining_slots]
            
            print(f"Depth {depth + 1}, Crawling {len(urls_to_crawl_this_level)} URLs: {list(urls_to_crawl_this_level)[:5]}...")

            # Define the retry logic outside the loop
            # Define the retry logic outside the loop
            @retry(
                wait=wait_exponential(multiplier=1, min=2, max=30),
                stop=stop_after_attempt(3),
                retry=retry_if_exception_type((ConnectionError, TimeoutError, requests.exceptions.RequestException))
            )
            async def crawl_with_retry(crawler, url_to_crawl):
                print(f"  -> Crawling: {url_to_crawl}")
                return await crawler.arun(url=url_to_crawl, config=crawl_config)

            for url in urls_to_crawl_this_level:
                if url not in visited:
                    visited.add(url)
                    # Pass crawler instance to the retry function
                    tasks.append(asyncio.create_task(crawl_with_retry(crawler, url)))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    print(f"WARN: Crawl task failed with exception: {result}")
                    continue

                if result and result.success:
                    if result.markdown:
                        results_all.append({'url': result.url, 'markdown': result.markdown})
                    
                    if limit is not None and len(results_all) >= limit:
                        print(f"Crawl limit ({limit}) reached. Stopping.")
                        current_urls = set()
                        break

                    if result.links:
                        internal_links = result.links.get("internal", [])
                        for link_info in internal_links:
                            href = link_info.get("href")
                            if href:
                                parsed_start_url = urlparse(start_urls[0])
                                parsed_link = urlparse(href)
                                if parsed_link.netloc == parsed_start_url.netloc:
                                    next_url, _ = urldefrag(href)
                                    if next_url not in visited:
                                        next_level_urls.add(next_url)
                    # --- End Link Extraction ---
            
            # If the inner loop broke due to limit, don't proceed to next level
            if not current_urls:
                break 

            current_urls = next_level_urls

    return results_all

async def crawl_batch(urls: List[str], max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """Batch crawl with retry and error skipping."""
    browser_config = BrowserConfig(headless=True, verbose=False)
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    
    results_all_cleaned = []
    
    # Define the retry logic once
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, requests.exceptions.RequestException))
    )
    async def crawl_single_url_with_retry(crawler, url):
        return await crawler.arun(url=url, config=crawl_config)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Create tasks inside the async context
        tasks = [asyncio.create_task(crawl_single_url_with_retry(crawler, url)) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"WARN: Failed to crawl {urls[i]} after multiple attempts. Error: {result}. Skipping.")
            elif result and result.success:
                if result.markdown:
                    results_all_cleaned.append({'url': result.url, 'markdown': result.markdown})
            elif result:
                 print(f"WARN: Crawl for {urls[i]} was not successful but did not raise an exception. Skipping.")

    return results_all_cleaned

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, requests.exceptions.RequestException))
)
async def crawl_markdown_file(url: str) -> List[Dict[str, Any]]:
    """Crawl a .txt or markdown file with retry."""
    browser_config = BrowserConfig(headless=True)
    crawl_config = CrawlerRunConfig()

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=crawl_config)
        if result.success and result.markdown:
            return [{'url': url, 'markdown': result.markdown}]
        elif result.error_message:
            raise Exception(f"Failed to crawl {url}: {result.error_message}")
        else:
            raise Exception(f"Failed to crawl {url} for an unknown reason.")

def parse_sitemap(sitemap_url: str) -> List[str]:
    resp = requests.get(sitemap_url)
    urls = []

    if resp.status_code == 200:
        try:
            tree = ElementTree.fromstring(resp.content)
            urls = [loc.text for loc in tree.findall('.//{*}loc')]
        except Exception as e:
            print(f"Error parsing sitemap XML: {e}")

    return urls

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
        name = f"{name}-kb" # kb for knowledge-base
    
    # Final check, if it's somehow empty after all this
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

async def run_ingestion(
    url: str,
    collection_name: Optional[str],
    db_dir: str,
    embedding_provider: str,
    embedding_model_name: Optional[str],
    vertex_project_id: Optional[str],
    vertex_location: Optional[str],
    chunk_size: int,
    chunk_overlap: int,
    max_depth: int,
    max_concurrent: int,
    batch_size: int,
    limit: Optional[int]
):
    """
    Refactored core logic of the ingestion process.
    This function can be called from Streamlit or other Python scripts.
    """
    try:
        # Sanitize and generate collection name
        if collection_name:
            collection_name = sanitize_collection_name(collection_name)
        else:
            collection_name = generate_collection_name_from_url(url)
        print(f"Using sanitized collection name: '{collection_name}'")

        # Set default embedding model name based on provider if not specified
        if embedding_model_name is None:
            if embedding_provider == "local":
                embedding_model_name = "intfloat/multilingual-e5-large"
            elif embedding_provider == "vertex_ai":
                embedding_model_name = "text-multilingual-embedding-002"

        print(f"Using Embedding Provider: {embedding_provider} with Model: {embedding_model_name}")

        # Detect URL type and crawl
        if is_txt(url):
            print(f"Detected .txt/markdown file: {url}")
            crawl_results = await crawl_markdown_file(url)
        elif is_sitemap(url):
            print(f"Detected sitemap: {url}")
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                raise ValueError("No URLs found in sitemap or sitemap could not be parsed.")
            crawl_results = await crawl_batch(sitemap_urls, max_concurrent=max_concurrent)
        else:
            print(f"Detected regular URL: {url}")
            crawl_results = await crawl_recursive_internal_links([url], max_depth=max_depth, max_concurrent=max_concurrent, limit=limit)

        # Chunk and collect metadata
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        # Initialize services/models based on provider
        local_embedding_model = None
        if embedding_provider == "local":
            print(f"Loading local embedding model '{embedding_model_name}'...")
            from sentence_transformers import SentenceTransformer
            local_embedding_model = SentenceTransformer(embedding_model_name)
            print("Local model loaded successfully.")
        elif embedding_provider == "vertex_ai":
            if not vertex_project_id:
                raise ValueError("--vertex-project-id is required when --embedding-provider is 'vertex_ai'")
            print(f"Initializing Vertex AI for project '{vertex_project_id}' in location '{vertex_location}'...")
            init_vertex_ai(project_id=vertex_project_id, location=vertex_location)
            print("Vertex AI initialized successfully.")

        # Collect documents
        for result in crawl_results:
            if result and result.get('markdown'):
                print(f"--- Processing URL: {result['url']} ---")
                markdown_content = result.get('markdown', '')
                chunks = smart_chunk_markdown(markdown_content, max_len=chunk_size, overlap_len=chunk_overlap)
                if not chunks and markdown_content:
                     print("WARNING: Markdown content present, but no chunks were generated.")
                if chunks:
                    for i, chunk_text in enumerate(chunks):
                        doc_id = f"doc_{result['url']}_{i}"
                        metadata = extract_section_info(chunk_text)
                        metadata['url'] = result['url']
                        all_chunks.append(chunk_text)
                        all_metadatas.append(metadata)
                        all_ids.append(doc_id)

        if not all_chunks:
            raise ValueError("Crawling finished, but no documents were found or could be processed to index.")

        # Generate embeddings before adding to ChromaDB
        all_embeddings = []
        if embedding_provider == "local":
            if local_embedding_model:
                print(f"Generating embeddings for {len(all_chunks)} chunks locally...")
                all_embeddings = local_embedding_model.encode(all_chunks, show_progress_bar=True).tolist()
                print("Embeddings generated successfully.")
        elif embedding_provider == "vertex_ai":
            print(f"Generating embeddings for {len(all_chunks)} chunks with Vertex AI ({embedding_model_name})...")
            all_embeddings_vertex = await get_vertex_text_embeddings_batched(
                texts=all_chunks,
                model_name=embedding_model_name,
                task_type="RETRIEVAL_DOCUMENT",
                project_id=vertex_project_id,
                location=vertex_location
            )
            # Check for failed embeddings and handle them gracefully
            failed_count = all_embeddings_vertex.count(None)
            if failed_count > 0:
                total_chunks = len(all_embeddings_vertex)
                failure_rate = failed_count / total_chunks
                
                print(f"WARNUNG: {failed_count} von {total_chunks} Embeddings fehlgeschlagen ({failure_rate:.1%})")
                
                # Only abort if failure rate is too high (more than 50%)
                if failure_rate > 0.5:
                    raise RuntimeError(
                        f"Zu viele Embedding-Fehler: {failed_count} von {total_chunks} Chunks ({failure_rate:.1%}). "
                        "Aborting ingestion to prevent incomplete knowledge base."
                    )
                else:
                    print(f"Fortsetzung mit {total_chunks - failed_count} erfolgreichen Embeddings...")
                    
                    # Filter out failed embeddings and their corresponding data
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
                    
                    # Update the lists to only include successful embeddings
                    all_chunks = filtered_chunks
                    all_metadatas = filtered_metadatas
                    all_ids = filtered_ids
                    all_embeddings = filtered_embeddings
                    
                    print(f"Gefilterte Daten: {len(all_chunks)} Chunks werden zur Datenbank hinzugefügt.")
            else:
                all_embeddings = all_embeddings_vertex

        print(f"Adding {len(all_chunks)} chunks to the ChromaDB collection '{collection_name}'...")
        chroma_client = get_chroma_client(persist_directory=db_dir)
        print(f"ChromaDB client created. Using collection: '{collection_name}'")

        # Define the embedding function based on the provider
        ef = None
        if embedding_provider == "local":
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model_name
            )

        collection = get_or_create_collection(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=ef,
            embedding_model_name=embedding_model_name
        )

        add_documents_to_collection(
            collection=collection,
            ids=all_ids,
            documents=all_chunks,
            embeddings=all_embeddings,
            metadatas=all_metadatas,
            batch_size=batch_size
        )
        print(f"Successfully added {len(all_chunks)} chunks to the ChromaDB collection '{collection_name}'.")

    except (requests.exceptions.RequestException, ValueError, RuntimeError) as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        # Re-raise the exception to be caught by the caller (e.g., Streamlit app)
        raise e
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        raise e

def main():
    parser = argparse.ArgumentParser(description="Insert crawled docs into ChromaDB using local or Vertex AI embeddings.")
    parser.add_argument("url", help="URL to crawl (regular, .txt, or sitemap)")
    parser.add_argument("--collection", default=None, help="Name of the ChromaDB collection. If not provided, a name will be generated from the URL.")
    parser.add_argument("--db-dir", default="./chroma_db", help="Directory where ChromaDB data is stored")
    parser.add_argument("--embedding-provider", default="vertex_ai", choices=["local", "vertex_ai"], help="Embedding provider to use. Defaults to 'vertex_ai'.")
    parser.add_argument("--embedding-model-name", default=None, help="Name of the embedding model. For 'local', a SentenceTransformer model name (e.g., 'intfloat/multilingual-e5-large'). For 'vertex_ai', a Vertex AI model identifier (e.g., 'text-multilingual-embedding-002', 'text-embedding-004'). Default wird basierend auf Provider gesetzt.")
    parser.add_argument("--vertex-project-id", default=os.getenv("GOOGLE_CLOUD_PROJECT"), help="Google Cloud Project ID for Vertex AI. Defaults to GOOGLE_CLOUD_PROJECT env var.")
    parser.add_argument("--vertex-location", default=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"), help="Google Cloud Location for Vertex AI. Defaults to GOOGLE_CLOUD_LOCATION env var or 'us-central1'.")
    parser.add_argument("--chunk-size", type=int, default=3000, help="Target chunk size in characters for smart_chunk_markdown.")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap size in characters for smart_chunk_markdown.")
    parser.add_argument("--max-depth", type=int, default=3, help="Max recursion depth for crawling (if not a sitemap or single file).")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Max parallel browser sessions")
    parser.add_argument("--batch-size", type=int, default=300, help="ChromaDB insert batch size")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of pages crawled (only for regular website crawl)")
    args = parser.parse_args()

    # Run the refactored logic using asyncio
    asyncio.run(run_ingestion(
        url=args.url,
        collection_name=args.collection,
        db_dir=args.db_dir,
        embedding_provider=args.embedding_provider,
        embedding_model_name=args.embedding_model_name,
        vertex_project_id=args.vertex_project_id,
        vertex_location=args.vertex_location,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_depth=args.max_depth,
        max_concurrent=args.max_concurrent,
        batch_size=args.batch_size,
        limit=args.limit
    ))


if __name__ == "__main__":
    main()
