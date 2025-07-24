"""
Intelligente URL-Typ-Erkennung für automatisches Crawling

Dieses Modul erkennt automatisch den besten Crawling-Typ basierend auf der URL
und stellt optimale Einstellungen bereit.
"""

import re
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class CrawlingMethod:
    """Datenklasse für erkannte Crawling-Methode."""
    method: str  # 'single', 'recursive', 'sitemap'
    icon: str
    description: str
    settings: Dict[str, Any]
    backend_method: str  # Welche crawler_client Methode verwenden


class URLDetector:
    """Intelligente URL-Typ-Erkennung."""
    
    def __init__(self):
        # Sitemap-Patterns
        self.sitemap_patterns = [
            r'sitemap\.xml$',
            r'sitemap_index\.xml$',
            r'sitemaps\.xml$',
            r'sitemap-index\.xml$',
            r'/sitemap\.xml$',
            r'/sitemap_index\.xml$',
            r'/sitemap/$',
            r'/sitemaps/$',
            r'sitemap\.xml\?',  # Mit Query-Parametern
            r'sitemap_index\.xml\?'
        ]
        
        # Einzelseiten-Patterns
        self.single_page_patterns = [
            r'\.html?$',
            r'\.php$',
            r'\.aspx?$',
            r'\.jsp$',
            r'\.pdf$',
            r'/[^/]+\.html?$',
            r'/[^/]+\.php$'
        ]
        
        # Domain/Website-Patterns (für recursive crawling)
        self.website_patterns = [
            r'^https?://[^/]+/?$',  # Nur Domain
            r'^https?://[^/]+/[^.]*/?$',  # Domain mit Pfad ohne Dateiendung
            r'docs?\.',  # Dokumentations-Subdomains
            r'help\.',
            r'support\.',
            r'knowledge\.',
            r'wiki\.',
            r'/docs?/',
            r'/help/',
            r'/support/',
            r'/wiki/',
            r'/knowledge/',
            r'/guide/',
            r'/tutorial/',
            r'/manual/'
        ]
    
    def detect_crawling_method(self, url: str) -> CrawlingMethod:
        """
        Erkennt automatisch die beste Crawling-Methode für eine URL.
        
        Args:
            url: Die zu analysierende URL
            
        Returns:
            CrawlingMethod mit allen Details
        """
        if not url or not url.strip():
            return CrawlingMethod(
                method="unknown",
                icon="❓",
                description="Keine URL angegeben",
                settings={},
                backend_method="single"
            )
        
        url = url.strip().lower()
        
        # 1. Sitemap Detection (höchste Priorität)
        if self._is_sitemap(url):
            return CrawlingMethod(
                method="sitemap",
                icon="🗺️",
                description="Sitemap wird automatisch geparst",
                settings={
                    "max_concurrent": 10,
                    "auto_detect_urls": True,
                    "recommended_reason": "Sitemap enthält alle wichtigen URLs der Website"
                },
                backend_method="sitemap"
            )
        
        # 2. Single Page Detection
        if self._is_single_page(url):
            return CrawlingMethod(
                method="single",
                icon="📄",
                description="Einzelne Seite wird gecrawlt",
                settings={
                    "cache_mode": "BYPASS",
                    "recommended_reason": "Spezifische Seite mit Dateiendung erkannt"
                },
                backend_method="single"
            )
        
        # 3. Documentation/Website Detection
        if self._is_documentation_site(url):
            return CrawlingMethod(
                method="documentation",
                icon="📚",
                description="Dokumentations-Website wird tiefgehend gecrawlt",
                settings={
                    "max_depth": 3,
                    "max_concurrent": 4,
                    "limit": 50,
                    "recommended_reason": "Dokumentations-Websites haben meist tiefe, strukturierte Inhalte"
                },
                backend_method="recursive"
            )
        
        # 4. Default: Website Recursive Crawling
        return CrawlingMethod(
            method="website",
            icon="🌐",
            description="Website wird rekursiv gecrawlt",
            settings={
                "max_depth": 2,
                "max_concurrent": 5,
                "limit": 20,
                "recommended_reason": "Standard-Einstellungen für Website-Crawling"
            },
            backend_method="recursive"
        )
    
    def _is_sitemap(self, url: str) -> bool:
        """Prüft ob URL eine Sitemap ist."""
        return any(re.search(pattern, url, re.IGNORECASE) for pattern in self.sitemap_patterns)
    
    def _is_single_page(self, url: str) -> bool:
        """Prüft ob URL eine einzelne Seite ist."""
        return any(re.search(pattern, url, re.IGNORECASE) for pattern in self.single_page_patterns)
    
    def _is_documentation_site(self, url: str) -> bool:
        """Prüft ob URL eine Dokumentations-Website ist."""
        return any(re.search(pattern, url, re.IGNORECASE) for pattern in self.website_patterns)
    
    def get_method_explanation(self, method: CrawlingMethod) -> str:
        """Gibt eine detaillierte Erklärung der erkannten Methode zurück."""
        explanations = {
            "sitemap": """
            **🗺️ Sitemap-Crawling:**
            - Alle URLs werden aus der Sitemap extrahiert
            - Sehr effizient für vollständige Website-Abdeckung
            - Keine Tiefenbegrenzung nötig
            - Optimal für große Websites
            """,
            "single": """
            **📄 Einzelseiten-Crawling:**
            - Nur die angegebene Seite wird verarbeitet
            - Schnell und ressourcenschonend
            - Ideal für spezifische Dokumente
            - Keine Link-Verfolgung
            """,
            "documentation": """
            **📚 Dokumentations-Crawling:**
            - Tieferes Crawling für strukturierte Inhalte
            - Folgt Navigationslinks und Unterseiten
            - Optimiert für Dokumentations-Websites
            - Höhere Seitenlimits
            """,
            "website": """
            **🌐 Website-Crawling:**
            - Rekursives Crawling der Website
            - Moderate Tiefe und Seitenzahl
            - Ausgewogene Einstellungen
            - Für allgemeine Websites
            """
        }
        
        return explanations.get(method.method, "Unbekannte Methode")
    
    def get_supported_url_types(self) -> str:
        """Gibt eine Übersicht der unterstützten URL-Typen zurück."""
        return """
        **💡 Unterstützte URL-Typen:**
        
        • **Website-URLs** → Rekursives Crawling
          `https://docs.example.com`, `https://example.com/help`
          
        • **Sitemap-URLs** → Automatisches Parsing  
          `https://example.com/sitemap.xml`, `https://site.com/sitemap_index.xml`
          
        • **Einzelseiten** → Direkte Extraktion
          `https://example.com/page.html`, `https://site.com/document.pdf`
          
        • **Dokumentations-Sites** → Tiefgehende Analyse
          `https://docs.example.com`, `https://help.example.com`
        """


# Globale Instanz für einfache Verwendung
url_detector = URLDetector()


# Convenience Functions
def detect_url_type(url: str) -> CrawlingMethod:
    """Convenience function für URL-Typ-Erkennung."""
    return url_detector.detect_crawling_method(url)


def get_optimal_settings(url: str) -> Dict[str, Any]:
    """Gibt optimale Crawling-Einstellungen für eine URL zurück."""
    method = detect_url_type(url)
    return method.settings


def get_backend_method(url: str) -> str:
    """Gibt die zu verwendende Backend-Methode zurück."""
    method = detect_url_type(url)
    return method.backend_method


# Test-Funktion
if __name__ == "__main__":
    # Test verschiedene URL-Typen
    test_urls = [
        "https://docs.python.org",
        "https://example.com/sitemap.xml",
        "https://example.com/page.html",
        "https://streamlit.io/docs",
        "https://github.com/streamlit/streamlit",
        "https://help.example.com",
        "https://example.com/sitemap_index.xml",
        "https://example.com/document.pdf"
    ]
    
    detector = URLDetector()
    
    print("🔍 URL-Typ-Erkennung Test")
    print("=" * 50)
    
    for url in test_urls:
        method = detector.detect_crawling_method(url)
        print(f"\n📍 URL: {url}")
        print(f"   {method.icon} Typ: {method.method}")
        print(f"   📝 Beschreibung: {method.description}")
        print(f"   🔧 Backend: {method.backend_method}")
        print(f"   ⚙️ Einstellungen: {method.settings}")
    
    print("\n✅ Test abgeschlossen!")