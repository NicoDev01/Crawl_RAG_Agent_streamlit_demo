"""
Modal.com Crawl4AI Service
--------------------------
Serverless crawling service using Crawl4AI and Playwright on Modal.com
"""

import modal
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from pydantic import BaseModel
from fastapi import Header, HTTPException
from typing import List, Optional, Dict, Any
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
import os
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree

# Definieren des Container-Images
crawler_image = modal.Image.debian_slim(python_version="3.11") \
    .pip_install_from_requirements("requirements.txt") \
    .run_commands(
        "playwright install-deps chromium",
        "playwright install chromium"
    )

# Initialisieren der Modal-App
app = modal.App("crawl4ai-service")

# Definieren der Request-Modelle
class CrawlSingleRequest(BaseModel):
    url: str
    cache_mode: Optional[str] = "BYPASS"

class CrawlBatchRequest(BaseModel):
    urls: List[str]
    max_concurrent: Optional[int] = 10

class CrawlRecursiveRequest(BaseModel):
    start_url: str
    max_depth: Optional[int] = 3
    max_concurrent: Optional[int] = 5
    limit: Optional[int] = 100

class CrawlSitemapRequest(BaseModel):
    sitemap_url: str
    max_concurrent: Optional[int] = 10

# Response-Modelle
class CrawlResult(BaseModel):
    success: bool
    url: str
    markdown: Optional[str] = None
    links: Optional[Dict] = None
    error: Optional[str] = None

class CrawlBatchResponse(BaseModel):
    results: List[CrawlResult]

# Hilfsfunktionen
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, requests.exceptions.RequestException))
)
async def crawl_with_retry(crawler, url, config):
    """Crawlt eine URL mit Retry-Logik für Netzwerkfehler."""
    print(f"  -> Crawling: {url}")
    return await crawler.arun(url=url, config=config)

def parse_sitemap(sitemap_url: str) -> List[str]:
    """Parst eine Sitemap-XML und extrahiert URLs."""
    try:
        resp = requests.get(sitemap_url, timeout=30)
        resp.raise_for_status()
        
        urls = []
        if resp.status_code == 200:
            try:
                tree = ElementTree.fromstring(resp.content)
                urls = [loc.text for loc in tree.findall('.//{*}loc')]
                print(f"Found {len(urls)} URLs in sitemap {sitemap_url}")
            except Exception as e:
                print(f"Error parsing sitemap XML: {e}")
        
        return urls
    except requests.exceptions.RequestException as e:
        print(f"Error fetching sitemap {sitemap_url}: {e}")
        return []

# API-Endpunkte
@app.function(
    image=crawler_image, 
    secrets=[modal.Secret.from_name("crawl4ai-api-key")],
    timeout=300  # 5 Minuten Timeout
)
@modal.web_endpoint(method="POST")
async def crawl_single(request: CrawlSingleRequest, authorization: str = Header(None)):
    """
    Crawlt eine einzelne URL und gibt strukturierte Daten zurück.
    
    Args:
        request: CrawlSingleRequest mit URL und optionalem cache_mode
        authorization: Bearer Token im Authorization Header
    
    Returns:
        CrawlResult mit success, url, markdown, links oder error
    """
    # API-Schlüssel-Validierung
    if not authorization or authorization != f"Bearer {os.environ['API_KEY']}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    print(f"Crawling single URL: {request.url}")
    
    try:
        # Browser- und Crawl-Konfiguration
        browser_config = BrowserConfig(headless=True, verbose=False)
        
        # Cache-Mode konfigurieren
        cache_mode = CacheMode.BYPASS
        if request.cache_mode and hasattr(CacheMode, request.cache_mode):
            cache_mode = getattr(CacheMode, request.cache_mode)
        
        crawl_config = CrawlerRunConfig(
            cache_mode=cache_mode,
            stream=False
        )
        
        # Crawling durchführen
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawl_with_retry(crawler, request.url, crawl_config)
            
            if result and result.success:
                return CrawlResult(
                    success=True,
                    url=result.url,
                    markdown=result.markdown,
                    links=result.links
                )
            else:
                error_msg = result.error_message if result else "Unknown crawling error"
                return CrawlResult(
                    success=False,
                    url=request.url,
                    error=error_msg
                )
                
    except Exception as e:
        error_message = f"Error during crawling: {str(e)}"
        print(f"ERROR: {error_message}")
        return CrawlResult(
            success=False,
            url=request.url,
            error=error_message
        )

# Batch Crawling Endpunkt
@app.function(
    image=crawler_image, 
    secrets=[modal.Secret.from_name("crawl4ai-api-key")],
    timeout=600  # 10 Minuten Timeout
)
@modal.web_endpoint(method="POST")
async def crawl_batch(request: CrawlBatchRequest, authorization: str = Header(None)):
    """
    Crawlt mehrere URLs parallel und gibt die Ergebnisse zurück.
    
    Args:
        request: CrawlBatchRequest mit URLs und optionalem max_concurrent
        authorization: Bearer Token im Authorization Header
    
    Returns:
        CrawlBatchResponse mit Liste von CrawlResults
    """
    # API-Schlüssel-Validierung
    if not authorization or authorization != f"Bearer {os.environ['API_KEY']}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    print(f"Batch crawling {len(request.urls)} URLs with max_concurrent={request.max_concurrent}")
    
    try:
        # Browser- und Crawl-Konfiguration
        browser_config = BrowserConfig(headless=True, verbose=False)
        crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        
        results_all = []
        
        # Crawling durchführen
        async with AsyncWebCrawler(config=browser_config) as crawler:
            # Erstelle Tasks für alle URLs
            tasks = [asyncio.create_task(crawl_with_retry(crawler, url, crawl_config)) for url in request.urls]
            
            # Führe Tasks parallel aus (mit Limit)
            semaphore = asyncio.Semaphore(request.max_concurrent)
            
            async def bounded_crawl(task, url):
                async with semaphore:
                    try:
                        return await task
                    except Exception as e:
                        print(f"Error crawling {url}: {e}")
                        return None
            
            bounded_tasks = [bounded_crawl(task, url) for task, url in zip(tasks, request.urls)]
            results = await asyncio.gather(*bounded_tasks)
            
            # Verarbeite Ergebnisse
            for i, result in enumerate(results):
                if result is None:
                    results_all.append(CrawlResult(
                        success=False,
                        url=request.urls[i],
                        error="Task failed with exception"
                    ))
                elif isinstance(result, Exception):
                    results_all.append(CrawlResult(
                        success=False,
                        url=request.urls[i],
                        error=str(result)
                    ))
                elif result and result.success:
                    results_all.append(CrawlResult(
                        success=True,
                        url=result.url,
                        markdown=result.markdown,
                        links=result.links
                    ))
                else:
                    error_msg = result.error_message if result else "Unknown crawling error"
                    results_all.append(CrawlResult(
                        success=False,
                        url=request.urls[i],
                        error=error_msg
                    ))
                    
        return CrawlBatchResponse(results=results_all)
        
    except Exception as e:
        error_message = f"Error during batch crawling: {str(e)}"
        print(f"ERROR: {error_message}")
        raise HTTPException(status_code=500, detail=error_message)

# Recursive Crawling Endpunkt
@app.function(
    image=crawler_image, 
    secrets=[modal.Secret.from_name("crawl4ai-api-key")],
    timeout=1200  # 20 Minuten Timeout
)
@modal.web_endpoint(method="POST")
async def crawl_recursive(request: CrawlRecursiveRequest, authorization: str = Header(None)):
    """
    Crawlt eine Website rekursiv und folgt internen Links bis zur angegebenen Tiefe.
    
    Args:
        request: CrawlRecursiveRequest mit start_url, max_depth, max_concurrent und limit
        authorization: Bearer Token im Authorization Header
    
    Returns:
        CrawlBatchResponse mit Liste von CrawlResults
    """
    # API-Schlüssel-Validierung
    if not authorization or authorization != f"Bearer {os.environ['API_KEY']}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    print(f"Recursive crawling from {request.start_url}, depth={request.max_depth}, limit={request.limit}")
    
    try:
        visited = set()
        current_urls = set([request.start_url])
        results_all = []
        browser_config = BrowserConfig(headless=True)
        crawl_config = CrawlerRunConfig()
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            for depth in range(request.max_depth):
                if not current_urls:
                    break
                
                # Check limit before starting new level
                if request.limit is not None and len(visited) >= request.limit:
                    break

                next_level_urls = set()
                tasks = []
                urls_to_crawl_this_level = list(current_urls - visited)
                
                # Apply limit within the current level as well
                if request.limit is not None:
                    remaining_slots = request.limit - len(visited)
                    urls_to_crawl_this_level = urls_to_crawl_this_level[:remaining_slots]
                
                print(f"Depth {depth + 1}, Crawling {len(urls_to_crawl_this_level)} URLs")

                # Semaphore für parallele Ausführung mit Limit
                semaphore = asyncio.Semaphore(request.max_concurrent)
                
                async def bounded_crawl(url):
                    async with semaphore:
                        try:
                            return await crawl_with_retry(crawler, url, crawl_config)
                        except Exception as e:
                            print(f"Error crawling {url}: {e}")
                            return None

                for url in urls_to_crawl_this_level:
                    if url not in visited:
                        visited.add(url)
                        tasks.append(asyncio.create_task(bounded_crawl(url)))

                results = await asyncio.gather(*tasks)
                
                for i, result in enumerate(results):
                    url = urls_to_crawl_this_level[i]
                    
                    if result is None:
                        continue  # Skip failed URLs
                        
                    if result and result.success:
                        results_all.append(CrawlResult(
                            success=True,
                            url=result.url,
                            markdown=result.markdown,
                            links=result.links
                        ))
                        
                        if request.limit is not None and len(results_all) >= request.limit:
                            current_urls = set()
                            break

                        # Extract internal links for next level
                        if result.links:
                            internal_links = result.links.get("internal", [])
                            for link_info in internal_links:
                                href = link_info.get("href")
                                if href:
                                    parsed_start_url = urlparse(request.start_url)
                                    parsed_link = urlparse(href)
                                    if parsed_link.netloc == parsed_start_url.netloc:
                                        next_url, _ = urldefrag(href)
                                        if next_url not in visited:
                                            next_level_urls.add(next_url)
                
                # If the inner loop broke due to limit, don't proceed to next level
                if not current_urls:
                    break 

                current_urls = next_level_urls

        return CrawlBatchResponse(results=results_all)
        
    except Exception as e:
        error_message = f"Error during recursive crawling: {str(e)}"
        print(f"ERROR: {error_message}")
        raise HTTPException(status_code=500, detail=error_message)

# Sitemap Crawling Endpunkt
@app.function(
    image=crawler_image, 
    secrets=[modal.Secret.from_name("crawl4ai-api-key")],
    timeout=900  # 15 Minuten Timeout
)
@modal.web_endpoint(method="POST")
async def crawl_sitemap(request: CrawlSitemapRequest, authorization: str = Header(None)):
    """
    Crawlt alle URLs aus einer Sitemap und gibt die Ergebnisse zurück.
    
    Args:
        request: CrawlSitemapRequest mit sitemap_url und optionalem max_concurrent
        authorization: Bearer Token im Authorization Header
    
    Returns:
        CrawlBatchResponse mit Liste von CrawlResults
    """
    # API-Schlüssel-Validierung
    if not authorization or authorization != f"Bearer {os.environ['API_KEY']}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    print(f"Crawling sitemap: {request.sitemap_url}")
    
    try:
        # Parse sitemap to get URLs
        sitemap_urls = parse_sitemap(request.sitemap_url)
        if not sitemap_urls:
            raise HTTPException(
                status_code=400, 
                detail="No URLs found in sitemap or sitemap could not be parsed."
            )
        
        print(f"Found {len(sitemap_urls)} URLs in sitemap")
        
        # Use batch crawling logic for sitemap URLs
        browser_config = BrowserConfig(headless=True, verbose=False)
        crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        
        results_all = []
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            # Semaphore für parallele Ausführung mit Limit
            semaphore = asyncio.Semaphore(request.max_concurrent)
            
            async def bounded_crawl(url):
                async with semaphore:
                    try:
                        return await crawl_with_retry(crawler, url, crawl_config)
                    except Exception as e:
                        print(f"Error crawling {url}: {e}")
                        return None
            
            # Erstelle Tasks für alle Sitemap URLs
            tasks = [asyncio.create_task(bounded_crawl(url)) for url in sitemap_urls]
            results = await asyncio.gather(*tasks)
            
            # Verarbeite Ergebnisse
            for i, result in enumerate(results):
                url = sitemap_urls[i]
                
                if result is None:
                    results_all.append(CrawlResult(
                        success=False,
                        url=url,
                        error="Task failed with exception"
                    ))
                elif isinstance(result, Exception):
                    results_all.append(CrawlResult(
                        success=False,
                        url=url,
                        error=str(result)
                    ))
                elif result and result.success:
                    results_all.append(CrawlResult(
                        success=True,
                        url=result.url,
                        markdown=result.markdown,
                        links=result.links
                    ))
                else:
                    error_msg = result.error_message if result else "Unknown crawling error"
                    results_all.append(CrawlResult(
                        success=False,
                        url=url,
                        error=error_msg
                    ))
                    
        return CrawlBatchResponse(results=results_all)
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        error_message = f"Error during sitemap crawling: {str(e)}"
        print(f"ERROR: {error_message}")
        raise HTTPException(status_code=500, detail=error_message)

# Test-Endpunkt für Debugging
@app.function(image=crawler_image)
@modal.web_endpoint(method="GET")
async def health_check():
    """Einfacher Health-Check Endpunkt."""
    return {"status": "healthy", "service": "crawl4ai-service"}

# Lokaler Test-Einstiegspunkt (optional)
@app.local_entrypoint()
def test_local():
    """Lokaler Test für die Entwicklung."""
    print("Testing crawl4ai service locally...")
    
    # Hier könntest du lokale Tests hinzufügen
    test_url = "https://example.com"
    print(f"Would crawl: {test_url}")
    print("Use 'modal deploy crawler_service.py' to deploy the service.")