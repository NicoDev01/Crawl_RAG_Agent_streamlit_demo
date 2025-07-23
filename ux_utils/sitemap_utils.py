"""
Sitemap Utilities for Streamlit RAG Knowledge Assistant

This module provides utilities for sitemap discovery, parsing, and analysis.
"""

import requests
import xml.etree.ElementTree as ET
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import re
from datetime import datetime


@dataclass
class SitemapURL:
    """Information about a URL found in a sitemap."""
    url: str
    lastmod: Optional[datetime] = None
    changefreq: Optional[str] = None
    priority: Optional[float] = None


@dataclass
class SitemapInfo:
    """Information about a discovered sitemap."""
    sitemap_url: str
    is_valid: bool
    url_count: int
    urls: List[SitemapURL]
    error_message: Optional[str] = None
    sitemap_type: str = "urlset"  # urlset or sitemapindex
    last_modified: Optional[datetime] = None
    file_size: Optional[int] = None


@dataclass
class SitemapDiscoveryResult:
    """Result of sitemap discovery process."""
    found_sitemaps: List[str]
    best_sitemap: Optional[SitemapInfo]
    total_urls: int
    discovery_method: str
    error_messages: List[str]


class SitemapUtils:
    """Utility class for sitemap operations."""
    
    def __init__(self, timeout: int = 15, user_agent: str = "RAG-Knowledge-Assistant/1.0"):
        self.timeout = timeout
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
    
    def discover_sitemap(self, base_url: str) -> SitemapDiscoveryResult:
        """
        Discover sitemaps for a given base URL using multiple strategies.
        
        Args:
            base_url: The base URL to search for sitemaps
            
        Returns:
            SitemapDiscoveryResult with discovered sitemaps
        """
        found_sitemaps = []
        error_messages = []
        
        # Normalize base URL
        parsed = urlparse(base_url)
        base_domain = f"{parsed.scheme}://{parsed.netloc}"
        
        # Common sitemap locations to check
        sitemap_paths = [
            "/sitemap.xml",
            "/sitemap_index.xml",
            "/sitemaps.xml",
            "/sitemap/sitemap.xml",
            "/sitemap/index.xml",
            "/wp-sitemap.xml",  # WordPress
            "/sitemap-index.xml",
            "/robots.txt"  # Check robots.txt for sitemap references
        ]
        
        # Strategy 1: Check common locations
        for path in sitemap_paths[:-1]:  # Exclude robots.txt for now
            sitemap_url = urljoin(base_domain, path)
            
            try:
                response = self.session.head(
                    sitemap_url,
                    timeout=self.timeout,
                    allow_redirects=True
                )
                
                if response.status_code == 200:
                    content_type = response.headers.get('content-type', '').lower()
                    if 'xml' in content_type or 'text' in content_type:
                        found_sitemaps.append(sitemap_url)
                
            except Exception as e:
                error_messages.append(f"Error checking {sitemap_url}: {str(e)}")
        
        # Strategy 2: Check robots.txt for sitemap references
        robots_sitemaps = self._check_robots_txt(base_domain)
        found_sitemaps.extend(robots_sitemaps)
        
        # Strategy 3: Try to infer from URL structure
        if not found_sitemaps:
            inferred_sitemaps = self._infer_sitemap_locations(base_url)
            found_sitemaps.extend(inferred_sitemaps)
        
        # Remove duplicates
        found_sitemaps = list(set(found_sitemaps))
        
        # Analyze found sitemaps to find the best one
        best_sitemap = None
        total_urls = 0
        
        if found_sitemaps:
            sitemap_analyses = []
            
            for sitemap_url in found_sitemaps:
                sitemap_info = self.parse_sitemap(sitemap_url)
                if sitemap_info.is_valid:
                    sitemap_analyses.append(sitemap_info)
                    total_urls += sitemap_info.url_count
            
            # Choose best sitemap (most URLs, or most recent)
            if sitemap_analyses:
                best_sitemap = max(sitemap_analyses, key=lambda s: s.url_count)
        
        # Determine discovery method
        if found_sitemaps:
            if any("/sitemap.xml" in url for url in found_sitemaps):
                discovery_method = "standard_location"
            elif robots_sitemaps:
                discovery_method = "robots_txt"
            else:
                discovery_method = "inferred"
        else:
            discovery_method = "none_found"
        
        return SitemapDiscoveryResult(
            found_sitemaps=found_sitemaps,
            best_sitemap=best_sitemap,
            total_urls=total_urls,
            discovery_method=discovery_method,
            error_messages=error_messages
        )
    
    def parse_sitemap(self, sitemap_url: str) -> SitemapInfo:
        """
        Parse a sitemap XML file and extract URL information.
        
        Args:
            sitemap_url: URL of the sitemap to parse
            
        Returns:
            SitemapInfo with parsed sitemap data
        """
        try:
            response = self.session.get(
                sitemap_url,
                timeout=self.timeout,
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Get file info
            file_size = len(response.content)
            last_modified = None
            
            last_mod_header = response.headers.get('last-modified')
            if last_mod_header:
                try:
                    last_modified = datetime.strptime(last_mod_header, '%a, %d %b %Y %H:%M:%S %Z')
                except:
                    pass
            
            # Parse XML
            try:
                root = ET.fromstring(response.content)
            except ET.ParseError as e:
                return SitemapInfo(
                    sitemap_url=sitemap_url,
                    is_valid=False,
                    url_count=0,
                    urls=[],
                    error_message=f"XML parsing error: {str(e)}",
                    file_size=file_size
                )
            
            # Determine sitemap type and namespace
            sitemap_type = "urlset"
            namespace = ""
            
            if root.tag.endswith('}urlset') or root.tag == 'urlset':
                sitemap_type = "urlset"
                if '}' in root.tag:
                    namespace = root.tag.split('}')[0] + '}'
            elif root.tag.endswith('}sitemapindex') or root.tag == 'sitemapindex':
                sitemap_type = "sitemapindex"
                if '}' in root.tag:
                    namespace = root.tag.split('}')[0] + '}'
            else:
                return SitemapInfo(
                    sitemap_url=sitemap_url,
                    is_valid=False,
                    url_count=0,
                    urls=[],
                    error_message=f"Unknown sitemap format: {root.tag}",
                    file_size=file_size
                )
            
            urls = []
            
            if sitemap_type == "urlset":
                # Parse URL entries
                for url_elem in root.findall(f'{namespace}url'):
                    loc_elem = url_elem.find(f'{namespace}loc')
                    if loc_elem is not None and loc_elem.text:
                        url_info = SitemapURL(url=loc_elem.text.strip())
                        
                        # Extract optional fields
                        lastmod_elem = url_elem.find(f'{namespace}lastmod')
                        if lastmod_elem is not None and lastmod_elem.text:
                            try:
                                url_info.lastmod = self._parse_date(lastmod_elem.text.strip())
                            except:
                                pass
                        
                        changefreq_elem = url_elem.find(f'{namespace}changefreq')
                        if changefreq_elem is not None and changefreq_elem.text:
                            url_info.changefreq = changefreq_elem.text.strip()
                        
                        priority_elem = url_elem.find(f'{namespace}priority')
                        if priority_elem is not None and priority_elem.text:
                            try:
                                url_info.priority = float(priority_elem.text.strip())
                            except:
                                pass
                        
                        urls.append(url_info)
            
            elif sitemap_type == "sitemapindex":
                # Parse sitemap index entries
                for sitemap_elem in root.findall(f'{namespace}sitemap'):
                    loc_elem = sitemap_elem.find(f'{namespace}loc')
                    if loc_elem is not None and loc_elem.text:
                        # For sitemap index, we store the sitemap URLs as regular URLs
                        # In a real implementation, you might want to recursively parse these
                        url_info = SitemapURL(url=loc_elem.text.strip())
                        
                        lastmod_elem = sitemap_elem.find(f'{namespace}lastmod')
                        if lastmod_elem is not None and lastmod_elem.text:
                            try:
                                url_info.lastmod = self._parse_date(lastmod_elem.text.strip())
                            except:
                                pass
                        
                        urls.append(url_info)
            
            return SitemapInfo(
                sitemap_url=sitemap_url,
                is_valid=True,
                url_count=len(urls),
                urls=urls,
                sitemap_type=sitemap_type,
                last_modified=last_modified,
                file_size=file_size
            )
            
        except requests.exceptions.RequestException as e:
            return SitemapInfo(
                sitemap_url=sitemap_url,
                is_valid=False,
                url_count=0,
                urls=[],
                error_message=f"Request error: {str(e)}"
            )
        except Exception as e:
            return SitemapInfo(
                sitemap_url=sitemap_url,
                is_valid=False,
                url_count=0,
                urls=[],
                error_message=f"Unexpected error: {str(e)}"
            )
    
    def validate_sitemap_content(self, sitemap_info: SitemapInfo) -> Dict[str, Any]:
        """
        Validate sitemap content and provide quality metrics.
        
        Args:
            sitemap_info: Parsed sitemap information
            
        Returns:
            Dictionary with validation results and metrics
        """
        if not sitemap_info.is_valid:
            return {
                "is_valid": False,
                "error": sitemap_info.error_message,
                "metrics": {}
            }
        
        metrics = {
            "total_urls": sitemap_info.url_count,
            "unique_urls": len(set(url.url for url in sitemap_info.urls)),
            "urls_with_lastmod": sum(1 for url in sitemap_info.urls if url.lastmod),
            "urls_with_priority": sum(1 for url in sitemap_info.urls if url.priority),
            "urls_with_changefreq": sum(1 for url in sitemap_info.urls if url.changefreq),
            "file_size_kb": sitemap_info.file_size / 1024 if sitemap_info.file_size else 0
        }
        
        # Check for common issues
        issues = []
        
        if metrics["total_urls"] != metrics["unique_urls"]:
            issues.append(f"Duplicate URLs found ({metrics['total_urls'] - metrics['unique_urls']} duplicates)")
        
        if metrics["total_urls"] > 50000:
            issues.append("Sitemap contains more than 50,000 URLs (consider splitting)")
        
        if metrics["file_size_kb"] > 10240:  # 10MB
            issues.append("Sitemap file is larger than 10MB (consider compression)")
        
        # Check URL patterns
        domains = set()
        protocols = set()
        
        for url in sitemap_info.urls:
            parsed = urlparse(url.url)
            domains.add(parsed.netloc)
            protocols.add(parsed.scheme)
        
        metrics["unique_domains"] = len(domains)
        metrics["protocols"] = list(protocols)
        
        if len(domains) > 1:
            issues.append(f"Multiple domains found in sitemap: {', '.join(domains)}")
        
        return {
            "is_valid": True,
            "metrics": metrics,
            "issues": issues,
            "quality_score": self._calculate_quality_score(metrics, issues)
        }
    
    def get_sitemap_statistics(self, sitemap_url: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a sitemap.
        
        Args:
            sitemap_url: URL of the sitemap
            
        Returns:
            Dictionary with sitemap statistics
        """
        sitemap_info = self.parse_sitemap(sitemap_url)
        validation_result = self.validate_sitemap_content(sitemap_info)
        
        return {
            "sitemap_info": sitemap_info,
            "validation": validation_result,
            "summary": {
                "is_valid": sitemap_info.is_valid,
                "url_count": sitemap_info.url_count,
                "sitemap_type": sitemap_info.sitemap_type,
                "file_size_kb": sitemap_info.file_size / 1024 if sitemap_info.file_size else 0,
                "quality_score": validation_result.get("quality_score", 0)
            }
        }
    
    def _check_robots_txt(self, base_domain: str) -> List[str]:
        """Check robots.txt for sitemap references."""
        robots_url = urljoin(base_domain, "/robots.txt")
        sitemaps = []
        
        try:
            response = self.session.get(robots_url, timeout=self.timeout)
            if response.status_code == 200:
                content = response.text
                
                # Find sitemap entries
                sitemap_matches = re.findall(r'^sitemap:\s*(.+)$', content, re.MULTILINE | re.IGNORECASE)
                for match in sitemap_matches:
                    sitemap_url = match.strip()
                    if sitemap_url.startswith('http'):
                        sitemaps.append(sitemap_url)
                    else:
                        sitemaps.append(urljoin(base_domain, sitemap_url))
        
        except Exception:
            pass
        
        return sitemaps
    
    def _infer_sitemap_locations(self, base_url: str) -> List[str]:
        """Infer possible sitemap locations based on URL structure."""
        # This is a placeholder for more sophisticated inference
        # Could analyze URL patterns, check for CMS-specific locations, etc.
        return []
    
    def _parse_date(self, date_string: str) -> datetime:
        """Parse various date formats found in sitemaps."""
        # Common sitemap date formats
        formats = [
            '%Y-%m-%d',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_string, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Unable to parse date: {date_string}")
    
    def _calculate_quality_score(self, metrics: Dict[str, Any], issues: List[str]) -> float:
        """Calculate a quality score for the sitemap (0.0 to 1.0)."""
        score = 1.0
        
        # Deduct points for issues
        score -= len(issues) * 0.1
        
        # Bonus for metadata completeness
        if metrics["total_urls"] > 0:
            lastmod_ratio = metrics["urls_with_lastmod"] / metrics["total_urls"]
            priority_ratio = metrics["urls_with_priority"] / metrics["total_urls"]
            changefreq_ratio = metrics["urls_with_changefreq"] / metrics["total_urls"]
            
            metadata_score = (lastmod_ratio + priority_ratio + changefreq_ratio) / 3
            score += metadata_score * 0.2
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))


# Convenience functions for direct use
def discover_sitemap(base_url: str, timeout: int = 15) -> SitemapDiscoveryResult:
    """Discover sitemaps (convenience function)."""
    utils = SitemapUtils(timeout=timeout)
    return utils.discover_sitemap(base_url)


def parse_sitemap(sitemap_url: str, timeout: int = 15) -> SitemapInfo:
    """Parse sitemap (convenience function)."""
    utils = SitemapUtils(timeout=timeout)
    return utils.parse_sitemap(sitemap_url)


def get_sitemap_statistics(sitemap_url: str, timeout: int = 15) -> Dict[str, Any]:
    """Get sitemap statistics (convenience function)."""
    utils = SitemapUtils(timeout=timeout)
    return utils.get_sitemap_statistics(sitemap_url)