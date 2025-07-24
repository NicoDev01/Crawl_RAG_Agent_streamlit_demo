"""
UX Components for Streamlit RAG Knowledge Assistant

This module contains reusable UI components for enhanced user experience
including URL validation, progress display, success animations, and more.
"""

import streamlit as st
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import asyncio
import urllib.parse
import requests
import time
import re
from abc import ABC, abstractmethod


# Data Models
@dataclass
class ValidationResult:
    """Result of URL validation with status and error information."""
    is_valid: bool
    error_message: Optional[str] = None
    status_code: Optional[int] = None
    response_time: Optional[float] = None
    warning_message: Optional[str] = None


class WebsiteType(Enum):
    """Enumeration of different website types for intelligent defaults."""
    DOCUMENTATION = "documentation"
    BLOG = "blog"
    NEWS = "news"
    CORPORATE = "corporate"
    ECOMMERCE = "ecommerce"
    UNKNOWN = "unknown"


@dataclass
class CrawlSettings:
    """Recommended crawling settings based on website type."""
    max_depth: int
    max_pages: int
    chunk_size: int
    max_concurrent: int
    recommended_reason: str


@dataclass
class LogEntry:
    """Log entry for detailed process tracking."""
    timestamp: datetime
    level: str  # INFO, WARNING, ERROR
    message: str
    url: Optional[str] = None
    details: Optional[Dict] = None


@dataclass
class CrawlResults:
    """Results from crawling process for success display."""
    documents_crawled: int
    chunks_created: int
    embeddings_generated: int
    total_time: float
    success_rate: float
    processing_speed: float


# Base Component Class
class BaseUXComponent(ABC):
    """Base class for all UX components with common functionality."""
    
    def __init__(self):
        self.component_id = self.__class__.__name__.lower()
    
    @abstractmethod
    def render(self) -> None:
        """Render the component in Streamlit."""
        pass
    
    def get_session_key(self, key: str) -> str:
        """Generate a unique session state key for this component."""
        return f"{self.component_id}_{key}"
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get component-specific session state."""
        session_key = self.get_session_key(key)
        return st.session_state.get(session_key, default)
    
    def set_state(self, key: str, value: Any) -> None:
        """Set component-specific session state."""
        session_key = self.get_session_key(key)
        st.session_state[session_key] = value


# URL Validation Component
class URLValidator(BaseUXComponent):
    """Component for real-time URL validation with visual feedback."""
    
    def __init__(self, timeout: int = 5, cache_ttl: int = 300, debounce_delay: float = 0.5):
        super().__init__()
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self.debounce_delay = debounce_delay
        self._last_validation_time = 0
    
    def validate_url_syntax(self, url: str) -> ValidationResult:
        """Validate URL syntax using urllib.parse."""
        if not url or not url.strip():
            return ValidationResult(
                is_valid=False,
                error_message="URL darf nicht leer sein"
            )
        
        try:
            parsed = urllib.parse.urlparse(url.strip())
            
            # Check for scheme and netloc
            if not parsed.scheme:
                return ValidationResult(
                    is_valid=False,
                    error_message="URL muss mit http:// oder https:// beginnen"
                )
            
            if parsed.scheme not in ['http', 'https']:
                return ValidationResult(
                    is_valid=False,
                    error_message="Nur HTTP und HTTPS URLs sind erlaubt"
                )
            
            if not parsed.netloc:
                return ValidationResult(
                    is_valid=False,
                    error_message="URL muss eine gültige Domain enthalten"
                )
            
            return ValidationResult(is_valid=True)
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Ungültige URL-Syntax: {str(e)}"
            )
    
    def check_url_reachability(self, url: str) -> ValidationResult:
        """Check if URL is reachable with HTTP HEAD request."""
        syntax_result = self.validate_url_syntax(url)
        if not syntax_result.is_valid:
            return syntax_result
        
        # Check cache first
        cache_key = f"url_check_{url}"
        cached_result = self.get_state(cache_key)
        if cached_result and self._is_cache_valid(cached_result):
            return cached_result['result']
        
        try:
            import time
            start_time = time.time()
            
            # Try HEAD request first (faster)
            try:
                response = requests.head(
                    url, 
                    timeout=self.timeout,
                    allow_redirects=True,
                    headers={'User-Agent': 'RAG-Knowledge-Assistant/1.0'}
                )
            except requests.exceptions.RequestException:
                # If HEAD fails, try GET with stream=True
                response = requests.get(
                    url,
                    timeout=self.timeout,
                    allow_redirects=True,
                    stream=True,
                    headers={'User-Agent': 'RAG-Knowledge-Assistant/1.0'}
                )
                # Close the connection immediately to avoid downloading content
                response.close()
            
            response_time = time.time() - start_time
            
            # Determine result based on status code
            if response.status_code == 200:
                result = ValidationResult(
                    is_valid=True,
                    status_code=response.status_code,
                    response_time=response_time
                )
                
                # Add performance warnings
                if response_time > 5.0:
                    result.warning_message = f"Sehr langsame Antwortzeit ({response_time:.1f}s)"
                elif response_time > 3.0:
                    result.warning_message = f"Langsame Antwortzeit ({response_time:.1f}s)"
                
            elif 300 <= response.status_code < 400:
                result = ValidationResult(
                    is_valid=True,
                    status_code=response.status_code,
                    response_time=response_time,
                    warning_message=f"URL leitet weiter (Status {response.status_code})"
                )
            elif response.status_code == 403:
                result = ValidationResult(
                    is_valid=False,
                    status_code=response.status_code,
                    response_time=response_time,
                    error_message="Zugriff verweigert (403 Forbidden) - möglicherweise Bot-Schutz"
                )
            elif response.status_code == 404:
                result = ValidationResult(
                    is_valid=False,
                    status_code=response.status_code,
                    response_time=response_time,
                    error_message="Seite nicht gefunden (404 Not Found)"
                )
            elif response.status_code == 429:
                result = ValidationResult(
                    is_valid=False,
                    status_code=response.status_code,
                    response_time=response_time,
                    error_message="Zu viele Anfragen (429 Rate Limited) - bitte später versuchen"
                )
            elif 500 <= response.status_code < 600:
                result = ValidationResult(
                    is_valid=False,
                    status_code=response.status_code,
                    response_time=response_time,
                    error_message=f"Server-Fehler (HTTP {response.status_code}) - Server temporär nicht verfügbar"
                )
            else:
                result = ValidationResult(
                    is_valid=False,
                    status_code=response.status_code,
                    response_time=response_time,
                    error_message=f"Unerwarteter HTTP Status: {response.status_code}"
                )
            
            # Cache the result
            self.set_state(cache_key, {
                'result': result,
                'timestamp': datetime.now()
            })
            
            return result
            
        except requests.exceptions.Timeout:
            return ValidationResult(
                is_valid=False,
                error_message=f"URL ist nicht erreichbar (Timeout nach {self.timeout}s)"
            )
        except requests.exceptions.ConnectionError:
            return ValidationResult(
                is_valid=False,
                error_message="Verbindung zur URL fehlgeschlagen - Server nicht erreichbar"
            )
        except requests.exceptions.TooManyRedirects:
            return ValidationResult(
                is_valid=False,
                error_message="Zu viele Weiterleitungen - möglicherweise Redirect-Loop"
            )
        except requests.exceptions.RequestException as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Netzwerk-Fehler: {str(e)}"
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Unerwarteter Fehler beim URL-Check: {str(e)}"
            )
    
    def _is_cache_valid(self, cached_data: Dict) -> bool:
        """Check if cached validation result is still valid."""
        if not cached_data or 'timestamp' not in cached_data:
            return False
        
        cache_age = (datetime.now() - cached_data['timestamp']).total_seconds()
        return cache_age < self.cache_ttl
    
    def _should_validate_now(self) -> bool:
        """Check if enough time has passed for debounced validation."""
        import time
        current_time = time.time()
        
        if current_time - self._last_validation_time >= self.debounce_delay:
            self._last_validation_time = current_time
            return True
        
        return False
    
    def validate_with_debounce(self, url: str, force: bool = False) -> Optional[ValidationResult]:
        """
        Validate URL with debouncing to avoid too frequent requests.
        
        Args:
            url: URL to validate
            force: Skip debounce delay if True
            
        Returns:
            ValidationResult if validation should run, None if debounced
        """
        if not url or not url.strip():
            return ValidationResult(is_valid=False, error_message="URL darf nicht leer sein")
        
        # Check debounce unless forced
        if not force and not self._should_validate_now():
            return None
        
        # First do syntax validation (fast)
        syntax_result = self.validate_url_syntax(url)
        if not syntax_result.is_valid:
            return syntax_result
        
        # Then do reachability check (slower)
        return self.check_url_reachability(url)
    
    def get_validation_indicator(self, result: ValidationResult) -> Tuple[str, str, str]:
        """Get visual indicator (icon, color, message) for validation result."""
        if not result.is_valid:
            return "🔴", "error", result.error_message or "URL ist ungültig"
        
        if result.warning_message:
            return "🟡", "warning", result.warning_message
        
        if result.response_time and result.response_time > 3.0:
            return "🟡", "warning", f"URL erreichbar, aber langsam ({result.response_time:.1f}s)"
        
        return "🟢", "success", "URL ist gültig und erreichbar - 👇 Passe die Crawling-Einstellungen unten an und klicke dann auf 'Erstellen'"
    
    def render_validation_feedback(self, url: str, show_reachability: bool = True, 
                                 container=None, debounced: bool = True) -> ValidationResult:
        """
        Render validation feedback for a URL input with enhanced visual indicators.
        
        Args:
            url: URL to validate
            show_reachability: Whether to check URL reachability
            container: Streamlit container to render in (optional)
            debounced: Whether to use debounced validation
            
        Returns:
            ValidationResult with validation details
        """
        if container is None:
            container = st
        
        if not url or not url.strip():
            return ValidationResult(is_valid=False)
        
        # Use debounced validation if requested
        if debounced:
            result = self.validate_with_debounce(url)
            if result is None:
                # Still debouncing, show loading state
                container.info("🔄 Validiere URL...")
                return ValidationResult(is_valid=False, error_message="Validierung läuft...")
        else:
            # Syntax validation first
            syntax_result = self.validate_url_syntax(url)
            if not syntax_result.is_valid:
                icon, color, message = self.get_validation_indicator(syntax_result)
                self._render_validation_message(container, color, icon, message)
                return syntax_result
            
            # Reachability check if requested
            if show_reachability:
                result = self.check_url_reachability(url)
            else:
                result = syntax_result
        
        # Render the validation result
        icon, color, message = self.get_validation_indicator(result)
        self._render_validation_message(container, color, icon, message, result)
        
        return result
    
    def _render_validation_message(self, container, color: str, icon: str, message: str, 
                                 result: ValidationResult = None) -> None:
        """Render validation message with enhanced styling."""
        
        if color == "error":
            container.error(f"{icon} **{message}**")
            
            # Add helpful tips for common errors
            if result and result.error_message:
                if "nicht erreichbar" in result.error_message:
                    container.caption("💡 Tipp: Prüfe ob die URL korrekt ist und der Server online ist")
                elif "Syntax" in result.error_message:
                    container.caption("💡 Tipp: URL muss mit http:// oder https:// beginnen")
                elif "403" in result.error_message:
                    container.caption("💡 Tipp: Diese Website blockiert möglicherweise automatisierte Zugriffe")
                elif "404" in result.error_message:
                    container.caption("💡 Tipp: Prüfe ob die URL korrekt geschrieben ist")
                    
        elif color == "warning":
            container.warning(f"{icon} **{message}**")
            
            # Add performance tips
            if result and result.response_time and result.response_time > 3.0:
                container.caption("⚡ Diese Website antwortet langsam - das Crawling könnte länger dauern")
                
        else:
            container.success(f"{icon} **{message}**")
            
            # Add success details
            if result and result.response_time:
                if result.response_time < 1.0:
                    container.caption(f"⚡ Schnelle Antwortzeit: {result.response_time:.2f}s")
                else:
                    container.caption(f"⏱️ Antwortzeit: {result.response_time:.2f}s")
    
    def render_realtime_validator(self, url_input_key: str = None, 
                                show_reachability: bool = True) -> Tuple[str, ValidationResult]:
        """
        Render a real-time URL validator with immediate feedback.
        
        Args:
            url_input_key: Custom key for the URL input field
            show_reachability: Whether to check URL reachability
            
        Returns:
            Tuple of (url, validation_result)
        """
        if url_input_key is None:
            url_input_key = self.get_session_key("url_input")
        
        # URL input field
        url = st.text_input(
            "Website URL:",
            key=url_input_key,
            placeholder="https://docs.example.com",
            help="Gib die vollständige URL der Website ein, die du crawlen möchtest"
        )
        
        # Validation feedback container
        validation_container = st.empty()
        
        if url and url.strip():
            with validation_container.container():
                result = self.render_validation_feedback(
                    url, 
                    show_reachability=show_reachability,
                    debounced=True
                )
                return url, result
        else:
            validation_container.empty()
            return "", ValidationResult(is_valid=False)
    
    def get_validation_status_indicator(self, result: ValidationResult) -> str:
        """Get a compact status indicator for use in other components."""
        if not result.is_valid:
            return "🔴"
        elif result.warning_message:
            return "🟡"
        else:
            return "🟢"
    
    def render(self) -> None:
        """Render the URL validator component (standalone demo)."""
        st.markdown("### 🔍 URL Validator Demo")
        
        url, result = self.render_realtime_validator()
        
        if url:
            # Show detailed validation info in expander
            with st.expander("🔧 Detaillierte Validierungsinfo", expanded=False):
                st.json({
                    "url": url,
                    "is_valid": result.is_valid,
                    "status_code": result.status_code,
                    "response_time": f"{result.response_time:.3f}s" if result.response_time else None,
                    "error_message": result.error_message,
                    "warning_message": result.warning_message
                })


# Process Display Component
class ProcessDisplay(BaseUXComponent):
    """Component for detailed progress tracking with expandable logging."""
    
    def __init__(self):
        super().__init__()
        self.reset_progress()
    
    def reset_progress(self) -> None:
        """Reset all progress tracking state."""
        self.set_state("current_step", 0)
        self.set_state("total_steps", 5)
        self.set_state("current_message", "")
        self.set_state("current_url", "")
        self.set_state("log_entries", [])
        self.set_state("statistics", {})
    
    def create_progress_container(self) -> None:
        """Create the main progress container."""
        if not hasattr(self, 'progress_container'):
            self.progress_container = st.container()
        
        with self.progress_container:
            # Main progress bar
            current_step = self.get_state("current_step", 0)
            total_steps = self.get_state("total_steps", 5)
            progress = min(current_step / total_steps, 1.0) if total_steps > 0 else 0
            
            self.progress_bar = st.progress(progress)
            self.status_text = st.empty()
            
            # Current URL display
            current_url = self.get_state("current_url", "")
            if current_url:
                st.info(f"🌐 Aktuell verarbeitet: {current_url}")
            
            # Statistics display
            self.render_live_statistics()
            
            # Expandable log
            self.render_expandable_log()
    
    def update_current_step(self, step: int, message: str, current_url: str = None) -> None:
        """Update the current processing step."""
        self.set_state("current_step", step)
        self.set_state("current_message", message)
        
        if current_url:
            self.set_state("current_url", current_url)
        
        # Update progress bar if it exists
        if hasattr(self, 'progress_bar'):
            total_steps = self.get_state("total_steps", 5)
            progress = min(step / total_steps, 1.0) if total_steps > 0 else 0
            self.progress_bar.progress(progress)
        
        # Update status text
        if hasattr(self, 'status_text'):
            self.status_text.text(f"Schritt {step}/{self.get_state('total_steps', 5)}: {message}")
    
    def add_log_entry(self, level: str, message: str, url: str = None, details: Dict = None) -> None:
        """Add a new log entry."""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            url=url,
            details=details
        )
        
        log_entries = self.get_state("log_entries", [])
        log_entries.append(entry)
        self.set_state("log_entries", log_entries)
    
    def render_expandable_log(self) -> None:
        """Render expandable detailed log."""
        log_entries = self.get_state("log_entries", [])
        
        if log_entries:
            with st.expander(f"📋 Detailliertes Log ({len(log_entries)} Einträge)", expanded=False):
                for entry in log_entries[-50:]:  # Show last 50 entries
                    timestamp_str = entry.timestamp.strftime("%H:%M:%S")
                    
                    if entry.level == "ERROR":
                        st.error(f"[{timestamp_str}] ❌ {entry.message}")
                    elif entry.level == "WARNING":
                        st.warning(f"[{timestamp_str}] ⚠️ {entry.message}")
                    else:
                        st.info(f"[{timestamp_str}] ℹ️ {entry.message}")
                    
                    if entry.url:
                        st.caption(f"   🔗 URL: {entry.url}")
    
    def render_live_statistics(self) -> None:
        """Render live processing statistics."""
        stats = self.get_state("statistics", {})
        
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("✅ Erfolgreich", stats.get("success_count", 0))
            
            with col2:
                st.metric("❌ Fehler", stats.get("error_count", 0))
            
            with col3:
                speed = stats.get("processing_speed", 0)
                st.metric("⚡ Geschwindigkeit", f"{speed:.1f}/min")
            
            with col4:
                elapsed = stats.get("elapsed_time", 0)
                st.metric("⏱️ Verstrichene Zeit", f"{elapsed:.0f}s")
    
    def update_statistics(self, stats: Dict[str, Any]) -> None:
        """Update processing statistics."""
        current_stats = self.get_state("statistics", {})
        current_stats.update(stats)
        self.set_state("statistics", current_stats)
    
    def render(self) -> None:
        """Render the process display component."""
        self.create_progress_container()


# Success Animation Component  
class SuccessAnimation(BaseUXComponent):
    """Component for success celebration and metrics display."""
    
    def trigger_success_celebration(self) -> None:
        """Trigger success celebration animation."""
        st.balloons()
        # Note: st.balloons() runs for ~3 seconds automatically
    
    def render_success_metrics_cards(self, results: CrawlResults) -> None:
        """Render success metrics in professional cards."""
        st.markdown("### 🎉 Erfolgreich abgeschlossen!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="📄 Dokumente gecrawlt",
                value=results.documents_crawled,
                help="Anzahl der erfolgreich verarbeiteten Webseiten"
            )
        
        with col2:
            st.metric(
                label="📝 Text-Chunks erstellt", 
                value=results.chunks_created,
                help="Anzahl der erstellten Textabschnitte für die Suche"
            )
        
        with col3:
            st.metric(
                label="🧠 Embeddings generiert",
                value=results.embeddings_generated,
                help="Anzahl der erstellten Vektor-Embeddings"
            )
        
        # Additional metrics row
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.metric(
                label="⏱️ Gesamtzeit",
                value=f"{results.total_time:.1f}s",
                help="Gesamte Verarbeitungszeit"
            )
        
        with col5:
            st.metric(
                label="✅ Erfolgsrate",
                value=f"{results.success_rate:.1%}",
                help="Prozentsatz erfolgreich verarbeiteter Seiten"
            )
        
        with col6:
            st.metric(
                label="⚡ Geschwindigkeit",
                value=f"{results.processing_speed:.1f}/min",
                help="Verarbeitungsgeschwindigkeit pro Minute"
            )
    
    def create_next_steps_cta(self) -> None:
        """Create call-to-action for next steps."""
        st.markdown("---")
        st.markdown("### 🚀 Nächste Schritte")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button(
                "💬 Jetzt mit der Wissensdatenbank chatten →",
                use_container_width=True,
                type="primary"
            ):
                # Switch to chat tab (this would need to be handled by parent component)
                st.session_state["switch_to_chat"] = True
                st.rerun()
    
    def show_completion_summary(self, results: CrawlResults) -> None:
        """Show expandable completion summary."""
        with st.expander("📊 Detaillierte Zusammenfassung", expanded=False):
            st.markdown(f"""
            **Verarbeitungsdetails:**
            - **Dokumente:** {results.documents_crawled} Webseiten erfolgreich gecrawlt
            - **Text-Chunks:** {results.chunks_created} Textabschnitte erstellt (durchschnittlich {results.chunks_created/max(results.documents_crawled, 1):.1f} pro Dokument)
            - **Embeddings:** {results.embeddings_generated} Vektor-Embeddings generiert
            - **Verarbeitungszeit:** {results.total_time:.1f} Sekunden
            - **Erfolgsrate:** {results.success_rate:.1%} der Seiten erfolgreich verarbeitet
            - **Durchschnittliche Geschwindigkeit:** {results.processing_speed:.1f} Seiten pro Minute
            
            **Qualitätsindikatoren:**
            - ✅ Alle Embeddings erfolgreich erstellt
            - ✅ Wissensdatenbank ist bereit für Abfragen
            - ✅ Optimale Chunk-Größe für präzise Antworten
            """)
    
    def render_complete_success_flow(self, results: CrawlResults) -> None:
        """Render the complete success flow with all components."""
        # Trigger celebration
        self.trigger_success_celebration()
        
        # Show metrics
        self.render_success_metrics_cards(results)
        
        # Show summary
        self.show_completion_summary(results)
        
        # Show next steps
        self.create_next_steps_cta()
    
    def render(self) -> None:
        """Render the success animation component."""
        # This would typically be called with actual results
        sample_results = CrawlResults(
            documents_crawled=10,
            chunks_created=45,
            embeddings_generated=45,
            total_time=120.5,
            success_rate=0.95,
            processing_speed=5.0
        )
        self.render_complete_success_flow(sample_results)


# Auto Complete Handler Component
class AutoCompleteHandler(BaseUXComponent):
    """Component for automatic database name completion from page titles."""
    
    def __init__(self, timeout: int = 10):
        super().__init__()
        self.timeout = timeout
    
    def extract_page_title(self, url: str) -> Optional[str]:
        """Extract page title from URL."""
        try:
            response = requests.get(
                url,
                timeout=self.timeout,
                headers={'User-Agent': 'RAG-Knowledge-Assistant/1.0'}
            )
            response.raise_for_status()
            
            # Simple title extraction (could be enhanced with BeautifulSoup)
            content = response.text
            start = content.find('<title>')
            end = content.find('</title>')
            
            if start != -1 and end != -1:
                title = content[start + 7:end].strip()
                return title
            
            return None
            
        except Exception:
            return None
    
    def generate_clean_name(self, title: str, url: str) -> str:
        """Generate clean database name from title and URL."""
        if title:
            # Remove common suffixes
            suffixes_to_remove = [
                "| Documentation", "- Documentation", "Documentation",
                "| Docs", "- Docs", "Docs",
                "| Home", "- Home", "Home",
                "| Welcome", "- Welcome", "Welcome"
            ]
            
            clean_title = title
            for suffix in suffixes_to_remove:
                if clean_title.endswith(suffix):
                    clean_title = clean_title[:-len(suffix)].strip()
            
            # Clean up the title
            clean_title = clean_title.replace("|", "-").replace("  ", " ").strip()
            
            if clean_title and len(clean_title) > 3:
                return clean_title
        
        # Fallback to domain-based name
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.replace("www.", "")
            return domain.split(".")[0].title()
        except:
            return "Wissensdatenbank"
    
    def get_fallback_names(self, url: str) -> List[str]:
        """Get fallback name suggestions based on URL."""
        suggestions = []
        
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.replace("www.", "")
            
            # Domain-based suggestions
            domain_parts = domain.split(".")
            if len(domain_parts) > 0:
                suggestions.append(domain_parts[0].title())
                suggestions.append(f"{domain_parts[0].title()} Docs")
                suggestions.append(f"{domain_parts[0].title()} Knowledge Base")
            
            # Path-based suggestions
            if parsed.path and parsed.path != "/":
                path_parts = [p for p in parsed.path.split("/") if p]
                if path_parts:
                    suggestions.append(path_parts[0].title().replace("-", " "))
        
        except:
            pass
        
        # Add generic suggestions
        suggestions.extend([
            "Dokumentation",
            "Wissensdatenbank", 
            "Knowledge Base"
        ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions[:5]  # Return top 5 suggestions
    
    def auto_fill_name_field(self, url: str) -> str:
        """Auto-fill name field based on URL analysis."""
        if not url:
            return ""
        
        # Try to extract title first
        title = self.extract_page_title(url)
        if title:
            return self.generate_clean_name(title, url)
        
        # Fallback to URL-based name
        fallbacks = self.get_fallback_names(url)
        return fallbacks[0] if fallbacks else "Wissensdatenbank"
    
    def render_name_suggestions(self, url: str, current_name: str = "") -> str:
        """Render name suggestions interface."""
        if not url:
            return current_name
        
        # Get suggestions
        suggestions = self.get_fallback_names(url)
        
        # Try to get title-based suggestion
        with st.spinner("🔍 Analysiere Webseite für Namensvorschläge..."):
            title_suggestion = self.auto_fill_name_field(url)
            if title_suggestion and title_suggestion not in suggestions:
                suggestions.insert(0, title_suggestion)
        
        if suggestions:
            st.markdown("💡 **Vorgeschlagene Namen:**")
            
            cols = st.columns(min(len(suggestions), 3))
            for i, suggestion in enumerate(suggestions[:3]):
                with cols[i]:
                    if st.button(f"📝 {suggestion}", key=f"name_suggestion_{i}"):
                        return suggestion
        
        return current_name
    
    def render(self) -> None:
        """Render the auto-complete handler component."""
        st.markdown("### Auto-Complete Name Handler")
        
        url = st.text_input("URL:", key=self.get_session_key("url"))
        current_name = st.text_input("Aktueller Name:", key=self.get_session_key("name"))
        
        if url:
            suggested_name = self.render_name_suggestions(url, current_name)
            if suggested_name != current_name:
                st.session_state[self.get_session_key("name")] = suggested_name
                st.rerun()


# Website Detector Component (Placeholder)
class WebsiteDetector(BaseUXComponent):
    """Component for intelligent website type detection."""
    
    def detect_website_type(self, url: str) -> WebsiteType:
        """Detect website type from URL patterns."""
        # This is a placeholder implementation
        # Would be implemented in a later task
        return WebsiteType.UNKNOWN
    
    def get_recommended_settings(self, website_type: WebsiteType) -> CrawlSettings:
        """Get recommended settings for website type."""
        # This is a placeholder implementation
        # Would be implemented in a later task
        return CrawlSettings(
            max_depth=2,
            max_pages=10,
            chunk_size=1200,
            max_concurrent=5,
            recommended_reason="Standard-Einstellungen"
        )
    
    def render(self) -> None:
        """Render the website detector component."""
        st.markdown("### Website Detector (Placeholder)")
        st.info("Diese Komponente wird in einer späteren Aufgabe implementiert.")


# Sitemap Detector Component (Placeholder)
class SitemapDetector(BaseUXComponent):
    """Component for automatic sitemap discovery."""
    
    def discover_sitemap(self, base_url: str) -> Optional[str]:
        """Discover sitemap URL from base URL."""
        # This is a placeholder implementation
        # Would be implemented in a later task
        return None
    
    def render(self) -> None:
        """Render the sitemap detector component."""
        st.markdown("### Sitemap Detector (Placeholder)")
        st.info("Diese Komponente wird in einer späteren Aufgabe implementiert.")