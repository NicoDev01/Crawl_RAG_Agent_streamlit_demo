"""
Web Scraper Utilities for Streamlit RAG Knowledge Assistant

This module provides utilities for web scraping, page title extraction,
and website type analysis.
"""

import requests
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import re
from urllib.parse import urljoin, urlparse
from enum import Enum


class WebsiteType(Enum):
    """Enumeration of different website types."""
    DOCUMENTATION = "documentation"
    BLOG = "blog"
    NEWS = "news"
    CORPORATE = "corporate"
    ECOMMERCE = "ecommerce"
    WIKI = "wiki"
    FORUM = "forum"
    UNKNOWN = "unknown"


@dataclass
class PageInfo:
    """Information extracted from a web page."""
    title: Optional[str] = None
    description: Optional[str] = None
    language: Optional[str] = None
    content_type: Optional[str] = None
    canonical_url: Optional[str] = None
    robots_meta: Optional[str] = None
    
    # SEO and structure info
    h1_tags: List[str] = None
    meta_keywords: Optional[str] = None
    
    # Technical info
    response_time: Optional[float] = None
    status_code: Optional[int] = None
    final_url: Optional[str] = None


@dataclass
class WebsiteAnalysis:
    """Analysis result for website type detection."""
    detected_type: WebsiteType
    confidence: float  # 0.0 to 1.0
    indicators: List[str]  # List of indicators that led to this classification
    recommended_settings: Dict[str, Any]


class WebScraper:
    """Web scraper for extracting page information and analyzing websites."""
    
    def __init__(self, timeout: int = 15, user_agent: str = "RAG-Knowledge-Assistant/1.0"):
        self.timeout = timeout
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
    
    def extract_page_info(self, url: str) -> PageInfo:
        """
        Extract comprehensive information from a web page.
        
        Args:
            url: The URL to scrape
            
        Returns:
            PageInfo object with extracted information
        """
        try:
            import time
            start_time = time.time()
            
            response = self.session.get(
                url,
                timeout=self.timeout,
                allow_redirects=True
            )
            
            response_time = time.time() - start_time
            response.raise_for_status()
            
            # Basic info
            page_info = PageInfo(
                response_time=response_time,
                status_code=response.status_code,
                final_url=response.url,
                content_type=response.headers.get('content-type', '').split(';')[0]
            )
            
            # Parse HTML content
            content = response.text
            
            # Extract title
            page_info.title = self._extract_title(content)
            
            # Extract meta description
            page_info.description = self._extract_meta_content(content, 'description')
            
            # Extract language
            page_info.language = self._extract_language(content)
            
            # Extract canonical URL
            page_info.canonical_url = self._extract_canonical_url(content, url)
            
            # Extract robots meta
            page_info.robots_meta = self._extract_meta_content(content, 'robots')
            
            # Extract H1 tags
            page_info.h1_tags = self._extract_h1_tags(content)
            
            # Extract meta keywords
            page_info.meta_keywords = self._extract_meta_content(content, 'keywords')
            
            return page_info
            
        except Exception as e:
            # Return minimal info with error details
            return PageInfo(
                status_code=getattr(e, 'response', {}).get('status_code'),
                final_url=url
            )
    
    def extract_page_title(self, url: str) -> Optional[str]:
        """
        Extract just the page title (optimized for speed).
        
        Args:
            url: The URL to scrape
            
        Returns:
            Page title or None if extraction fails
        """
        try:
            # Use HEAD request first to check if it's HTML
            head_response = self.session.head(url, timeout=5, allow_redirects=True)
            content_type = head_response.headers.get('content-type', '').lower()
            
            if 'text/html' not in content_type:
                return None
            
            # Get partial content for title extraction
            response = self.session.get(
                url,
                timeout=self.timeout,
                allow_redirects=True,
                stream=True
            )
            
            # Read only first 8KB (usually enough for title)
            content = ""
            for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                content += chunk
                if len(content) > 8192 or '</title>' in content.lower():
                    break
            
            return self._extract_title(content)
            
        except Exception:
            return None
    
    def analyze_website_type(self, url: str, page_info: Optional[PageInfo] = None) -> WebsiteAnalysis:
        """
        Analyze website type based on URL patterns and page content.
        
        Args:
            url: The URL to analyze
            page_info: Optional pre-extracted page info
            
        Returns:
            WebsiteAnalysis with detected type and confidence
        """
        indicators = []
        confidence_scores = {}
        
        # URL-based analysis
        url_lower = url.lower()
        domain = urlparse(url).netloc.lower()
        
        # Documentation indicators
        doc_patterns = [
            r'docs?\.',
            r'documentation',
            r'manual',
            r'guide',
            r'api\.',
            r'developer',
            r'reference'
        ]
        
        for pattern in doc_patterns:
            if re.search(pattern, url_lower):
                indicators.append(f"URL contains '{pattern}'")
                confidence_scores[WebsiteType.DOCUMENTATION] = confidence_scores.get(WebsiteType.DOCUMENTATION, 0) + 0.3
        
        # Blog indicators
        blog_patterns = [
            r'blog',
            r'news',
            r'article',
            r'post',
            r'medium\.com',
            r'wordpress',
            r'blogspot'
        ]
        
        for pattern in blog_patterns:
            if re.search(pattern, url_lower):
                indicators.append(f"URL contains '{pattern}'")
                confidence_scores[WebsiteType.BLOG] = confidence_scores.get(WebsiteType.BLOG, 0) + 0.3
        
        # Wiki indicators
        wiki_patterns = [
            r'wiki',
            r'confluence',
            r'notion\.so'
        ]
        
        for pattern in wiki_patterns:
            if re.search(pattern, url_lower):
                indicators.append(f"URL contains '{pattern}'")
                confidence_scores[WebsiteType.WIKI] = confidence_scores.get(WebsiteType.WIKI, 0) + 0.4
        
        # E-commerce indicators
        ecommerce_patterns = [
            r'shop',
            r'store',
            r'cart',
            r'product',
            r'buy',
            r'amazon\.',
            r'ebay\.'
        ]
        
        for pattern in ecommerce_patterns:
            if re.search(pattern, url_lower):
                indicators.append(f"URL contains '{pattern}'")
                confidence_scores[WebsiteType.ECOMMERCE] = confidence_scores.get(WebsiteType.ECOMMERCE, 0) + 0.3
        
        # Content-based analysis (if page_info provided)
        if page_info:
            # Title analysis
            if page_info.title:
                title_lower = page_info.title.lower()
                
                if any(word in title_lower for word in ['documentation', 'docs', 'api', 'reference', 'guide']):
                    indicators.append("Title suggests documentation")
                    confidence_scores[WebsiteType.DOCUMENTATION] = confidence_scores.get(WebsiteType.DOCUMENTATION, 0) + 0.2
                
                if any(word in title_lower for word in ['blog', 'news', 'article']):
                    indicators.append("Title suggests blog/news")
                    confidence_scores[WebsiteType.BLOG] = confidence_scores.get(WebsiteType.BLOG, 0) + 0.2
                
                if 'wiki' in title_lower:
                    indicators.append("Title suggests wiki")
                    confidence_scores[WebsiteType.WIKI] = confidence_scores.get(WebsiteType.WIKI, 0) + 0.3
            
            # H1 tags analysis
            if page_info.h1_tags:
                h1_text = ' '.join(page_info.h1_tags).lower()
                
                if any(word in h1_text for word in ['getting started', 'installation', 'quickstart', 'tutorial']):
                    indicators.append("H1 tags suggest documentation")
                    confidence_scores[WebsiteType.DOCUMENTATION] = confidence_scores.get(WebsiteType.DOCUMENTATION, 0) + 0.15
        
        # Determine final type and confidence
        if confidence_scores:
            detected_type = max(confidence_scores.keys(), key=lambda k: confidence_scores[k])
            confidence = min(confidence_scores[detected_type], 1.0)
        else:
            detected_type = WebsiteType.UNKNOWN
            confidence = 0.0
            indicators.append("No clear indicators found")
        
        # Generate recommended settings
        recommended_settings = self._get_recommended_settings(detected_type)
        
        return WebsiteAnalysis(
            detected_type=detected_type,
            confidence=confidence,
            indicators=indicators,
            recommended_settings=recommended_settings
        )
    
    def _extract_title(self, content: str) -> Optional[str]:
        """Extract title from HTML content."""
        # Try to find title tag
        title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = title_match.group(1).strip()
            # Clean up HTML entities and extra whitespace
            title = re.sub(r'&[a-zA-Z0-9#]+;', ' ', title)
            title = re.sub(r'\s+', ' ', title).strip()
            return title if title else None
        
        return None
    
    def _extract_meta_content(self, content: str, name: str) -> Optional[str]:
        """Extract meta tag content by name."""
        patterns = [
            rf'<meta\s+name=["\']?{name}["\']?\s+content=["\']([^"\']*)["\'][^>]*>',
            rf'<meta\s+content=["\']([^"\']*)["\']?\s+name=["\']?{name}["\'][^>]*>'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_language(self, content: str) -> Optional[str]:
        """Extract language from HTML content."""
        # Try html lang attribute
        lang_match = re.search(r'<html[^>]*lang=["\']?([^"\'>\s]+)', content, re.IGNORECASE)
        if lang_match:
            return lang_match.group(1)
        
        # Try meta http-equiv
        meta_lang = self._extract_meta_content(content, 'language')
        if meta_lang:
            return meta_lang
        
        return None
    
    def _extract_canonical_url(self, content: str, base_url: str) -> Optional[str]:
        """Extract canonical URL from HTML content."""
        canonical_match = re.search(r'<link[^>]*rel=["\']?canonical["\']?[^>]*href=["\']([^"\']*)["\']', content, re.IGNORECASE)
        if canonical_match:
            canonical_url = canonical_match.group(1)
            # Make absolute URL if relative
            return urljoin(base_url, canonical_url)
        
        return None
    
    def _extract_h1_tags(self, content: str) -> List[str]:
        """Extract all H1 tag contents."""
        h1_matches = re.findall(r'<h1[^>]*>(.*?)</h1>', content, re.IGNORECASE | re.DOTALL)
        h1_tags = []
        
        for match in h1_matches:
            # Clean HTML tags and entities
            clean_text = re.sub(r'<[^>]+>', '', match)
            clean_text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', clean_text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            if clean_text:
                h1_tags.append(clean_text)
        
        return h1_tags
    
    def _get_recommended_settings(self, website_type: WebsiteType) -> Dict[str, Any]:
        """Get recommended crawling settings for website type."""
        settings_map = {
            WebsiteType.DOCUMENTATION: {
                "max_depth": 3,
                "max_pages": 50,
                "chunk_size": 1500,
                "max_concurrent": 3,
                "reason": "Dokumentations-Websites haben meist tiefe Strukturen mit viel zusammenhängendem Inhalt"
            },
            WebsiteType.BLOG: {
                "max_depth": 2,
                "max_pages": 20,
                "chunk_size": 1200,
                "max_concurrent": 5,
                "reason": "Blog-Artikel sind meist eigenständig und benötigen moderate Einstellungen"
            },
            WebsiteType.NEWS: {
                "max_depth": 2,
                "max_pages": 15,
                "chunk_size": 1000,
                "max_concurrent": 5,
                "reason": "News-Artikel sind kurz und aktuell, moderate Tiefe reicht aus"
            },
            WebsiteType.WIKI: {
                "max_depth": 3,
                "max_pages": 40,
                "chunk_size": 1400,
                "max_concurrent": 4,
                "reason": "Wiki-Seiten sind stark verlinkt und enthalten strukturiertes Wissen"
            },
            WebsiteType.CORPORATE: {
                "max_depth": 2,
                "max_pages": 10,
                "chunk_size": 1200,
                "max_concurrent": 5,
                "reason": "Unternehmens-Websites haben meist flache Strukturen mit wenigen Seiten"
            },
            WebsiteType.ECOMMERCE: {
                "max_depth": 1,
                "max_pages": 5,
                "chunk_size": 800,
                "max_concurrent": 3,
                "reason": "E-Commerce-Seiten enthalten viele Produktdaten, die meist nicht relevant sind"
            },
            WebsiteType.FORUM: {
                "max_depth": 2,
                "max_pages": 25,
                "chunk_size": 1000,
                "max_concurrent": 4,
                "reason": "Forum-Threads enthalten Diskussionen, moderate Einstellungen sind optimal"
            },
            WebsiteType.UNKNOWN: {
                "max_depth": 2,
                "max_pages": 10,
                "chunk_size": 1200,
                "max_concurrent": 5,
                "reason": "Konservative Standard-Einstellungen für unbekannte Website-Typen"
            }
        }
        
        return settings_map.get(website_type, settings_map[WebsiteType.UNKNOWN])


# Convenience functions for direct use
def extract_page_title(url: str, timeout: int = 15) -> Optional[str]:
    """Extract page title (convenience function)."""
    scraper = WebScraper(timeout=timeout)
    return scraper.extract_page_title(url)


def extract_page_info(url: str, timeout: int = 15) -> PageInfo:
    """Extract page information (convenience function)."""
    scraper = WebScraper(timeout=timeout)
    return scraper.extract_page_info(url)


def analyze_website_type(url: str, timeout: int = 15) -> WebsiteAnalysis:
    """Analyze website type (convenience function)."""
    scraper = WebScraper(timeout=timeout)
    page_info = scraper.extract_page_info(url)
    return scraper.analyze_website_type(url, page_info)