"""
URL Utilities for Streamlit RAG Knowledge Assistant

This module provides utilities for URL validation, analysis, and manipulation.
"""

import urllib.parse
import requests
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import time
import re


@dataclass
class URLValidationResult:
    """Result of URL validation with detailed information."""
    is_valid: bool
    error_message: Optional[str] = None
    warning_message: Optional[str] = None
    status_code: Optional[int] = None
    response_time: Optional[float] = None
    final_url: Optional[str] = None  # After redirects
    content_type: Optional[str] = None
    content_length: Optional[int] = None


class URLUtils:
    """Utility class for URL operations."""
    
    def __init__(self, timeout: int = 10, user_agent: str = "RAG-Knowledge-Assistant/1.0"):
        self.timeout = timeout
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
    
    def validate_url_syntax(self, url: str) -> URLValidationResult:
        """
        Validate URL syntax using urllib.parse.
        
        Args:
            url: The URL to validate
            
        Returns:
            URLValidationResult with validation details
        """
        if not url or not url.strip():
            return URLValidationResult(
                is_valid=False,
                error_message="URL darf nicht leer sein"
            )
        
        url = url.strip()
        
        try:
            parsed = urllib.parse.urlparse(url)
            
            # Check for scheme
            if not parsed.scheme:
                return URLValidationResult(
                    is_valid=False,
                    error_message="URL muss mit http:// oder https:// beginnen"
                )
            
            # Check scheme type
            if parsed.scheme.lower() not in ['http', 'https']:
                return URLValidationResult(
                    is_valid=False,
                    error_message="Nur HTTP und HTTPS URLs sind erlaubt"
                )
            
            # Check for netloc (domain)
            if not parsed.netloc:
                return URLValidationResult(
                    is_valid=False,
                    error_message="URL muss eine gültige Domain enthalten"
                )
            
            # Check for valid domain format
            domain_pattern = re.compile(
                r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
            )
            
            domain = parsed.netloc.split(':')[0]  # Remove port if present
            if not domain_pattern.match(domain):
                return URLValidationResult(
                    is_valid=False,
                    error_message="Ungültiges Domain-Format"
                )
            
            # Check for suspicious patterns
            suspicious_patterns = [
                r'localhost',
                r'127\.0\.0\.1',
                r'192\.168\.',
                r'10\.',
                r'172\.(1[6-9]|2[0-9]|3[0-1])\.'
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, domain, re.IGNORECASE):
                    return URLValidationResult(
                        is_valid=True,
                        warning_message="Lokale oder private IP-Adresse erkannt"
                    )
            
            return URLValidationResult(is_valid=True)
            
        except Exception as e:
            return URLValidationResult(
                is_valid=False,
                error_message=f"Ungültige URL-Syntax: {str(e)}"
            )
    
    def check_url_reachability(self, url: str, method: str = "HEAD") -> URLValidationResult:
        """
        Check if URL is reachable with HTTP request.
        
        Args:
            url: The URL to check
            method: HTTP method to use (HEAD or GET)
            
        Returns:
            URLValidationResult with reachability details
        """
        # First validate syntax
        syntax_result = self.validate_url_syntax(url)
        if not syntax_result.is_valid:
            return syntax_result
        
        try:
            start_time = time.time()
            
            # Try HEAD request first (faster)
            if method.upper() == "HEAD":
                response = self.session.head(
                    url,
                    timeout=self.timeout,
                    allow_redirects=True
                )
            else:
                response = self.session.get(
                    url,
                    timeout=self.timeout,
                    allow_redirects=True,
                    stream=True  # Don't download full content
                )
            
            response_time = time.time() - start_time
            
            # Get content info
            content_type = response.headers.get('content-type', '').split(';')[0]
            content_length = response.headers.get('content-length')
            content_length = int(content_length) if content_length else None
            
            # Determine result based on status code
            if response.status_code == 200:
                result = URLValidationResult(
                    is_valid=True,
                    status_code=response.status_code,
                    response_time=response_time,
                    final_url=response.url,
                    content_type=content_type,
                    content_length=content_length
                )
                
                # Add warnings for slow responses
                if response_time > 5.0:
                    result.warning_message = f"Langsame Antwortzeit ({response_time:.1f}s)"
                elif response_time > 3.0:
                    result.warning_message = f"Mäßige Antwortzeit ({response_time:.1f}s)"
                
                return result
                
            elif 300 <= response.status_code < 400:
                return URLValidationResult(
                    is_valid=True,
                    status_code=response.status_code,
                    response_time=response_time,
                    final_url=response.url,
                    content_type=content_type,
                    warning_message=f"URL leitet weiter (Status {response.status_code})"
                )
                
            elif response.status_code == 403:
                return URLValidationResult(
                    is_valid=False,
                    status_code=response.status_code,
                    response_time=response_time,
                    error_message="Zugriff verweigert (403 Forbidden)"
                )
                
            elif response.status_code == 404:
                return URLValidationResult(
                    is_valid=False,
                    status_code=response.status_code,
                    response_time=response_time,
                    error_message="Seite nicht gefunden (404 Not Found)"
                )
                
            elif response.status_code == 429:
                return URLValidationResult(
                    is_valid=False,
                    status_code=response.status_code,
                    response_time=response_time,
                    error_message="Zu viele Anfragen (429 Rate Limited)"
                )
                
            elif 500 <= response.status_code < 600:
                return URLValidationResult(
                    is_valid=False,
                    status_code=response.status_code,
                    response_time=response_time,
                    error_message=f"Server-Fehler (HTTP {response.status_code})"
                )
                
            else:
                return URLValidationResult(
                    is_valid=False,
                    status_code=response.status_code,
                    response_time=response_time,
                    error_message=f"Unerwarteter HTTP Status: {response.status_code}"
                )
                
        except requests.exceptions.Timeout:
            return URLValidationResult(
                is_valid=False,
                error_message=f"Timeout nach {self.timeout} Sekunden"
            )
            
        except requests.exceptions.ConnectionError:
            return URLValidationResult(
                is_valid=False,
                error_message="Verbindung zur URL fehlgeschlagen"
            )
            
        except requests.exceptions.TooManyRedirects:
            return URLValidationResult(
                is_valid=False,
                error_message="Zu viele Weiterleitungen"
            )
            
        except requests.exceptions.RequestException as e:
            return URLValidationResult(
                is_valid=False,
                error_message=f"Netzwerk-Fehler: {str(e)}"
            )
            
        except Exception as e:
            return URLValidationResult(
                is_valid=False,
                error_message=f"Unerwarteter Fehler: {str(e)}"
            )
    
    def normalize_url(self, url: str) -> str:
        """
        Normalize URL by removing fragments, sorting query parameters, etc.
        
        Args:
            url: The URL to normalize
            
        Returns:
            Normalized URL string
        """
        try:
            parsed = urllib.parse.urlparse(url.strip())
            
            # Remove fragment
            normalized = urllib.parse.urlunparse((
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path,
                parsed.params,
                parsed.query,
                ''  # Remove fragment
            ))
            
            return normalized
            
        except Exception:
            return url.strip()
    
    def extract_domain(self, url: str) -> Optional[str]:
        """
        Extract domain from URL.
        
        Args:
            url: The URL to extract domain from
            
        Returns:
            Domain string or None if invalid
        """
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Remove port if present
            domain = domain.split(':')[0]
            
            return domain
            
        except Exception:
            return None
    
    def is_same_domain(self, url1: str, url2: str) -> bool:
        """
        Check if two URLs are from the same domain.
        
        Args:
            url1: First URL
            url2: Second URL
            
        Returns:
            True if same domain, False otherwise
        """
        domain1 = self.extract_domain(url1)
        domain2 = self.extract_domain(url2)
        
        return domain1 is not None and domain1 == domain2
    
    def get_base_url(self, url: str) -> Optional[str]:
        """
        Get base URL (scheme + netloc) from full URL.
        
        Args:
            url: The full URL
            
        Returns:
            Base URL string or None if invalid
        """
        try:
            parsed = urllib.parse.urlparse(url)
            return f"{parsed.scheme}://{parsed.netloc}"
        except Exception:
            return None


# Convenience functions for direct use
def validate_url_syntax(url: str) -> URLValidationResult:
    """Validate URL syntax (convenience function)."""
    utils = URLUtils()
    return utils.validate_url_syntax(url)


def check_url_reachability(url: str, timeout: int = 10) -> URLValidationResult:
    """Check URL reachability (convenience function)."""
    utils = URLUtils(timeout=timeout)
    return utils.check_url_reachability(url)


def normalize_url(url: str) -> str:
    """Normalize URL (convenience function)."""
    utils = URLUtils()
    return utils.normalize_url(url)


def extract_domain(url: str) -> Optional[str]:
    """Extract domain from URL (convenience function)."""
    utils = URLUtils()
    return utils.extract_domain(url)